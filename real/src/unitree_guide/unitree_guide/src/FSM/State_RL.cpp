/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include <iostream>
#include "FSM/State_RL.h"

#include <fstream>
void saveArrayToFile(const float actions[], size_t size, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        for (size_t i = 0; i < size; ++i) {
            file << actions[i];
            if (i < size - 1) file << " ";
        }
        file << "\n";
        file.close();
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

State_RL::State_RL(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::RL, "rl"), 
    _est(ctrlComp->estimator), _contact(ctrlComp->contact), 
    _robModel(ctrlComp->robotModel){
        this->_vxLim = _robModel->getRobVelLimitX();
        this->_vyLim = _robModel->getRobVelLimitY();
        this->_wyawLim = _robModel->getRobVelLimitYaw();
}

void State_RL::_init_buffers()
{
    this->_dYawCmdPast=0.0;

    this->_action = torch::zeros({1, 12});
    this->_observation = torch::zeros({1, this->_num_obs});
    this->_gravity_vec = torch::tensor({{0.0, 0.0, -1.0}});
    this->_obs_buffer_tensor = torch::zeros({1, this->_num_obs_history*this->_num_obs});
}

void State_RL::_loadPolicy()  // 加载JIT模型
{
    this->_body_module = torch::jit::load(BODY_MODEL_PATH);
    // this->_adapt_module = torch::jit::load(ADAPT_MODEL_PATH);
}

torch::Tensor State_RL::QuatRotateInverse(torch::Tensor q, torch::Tensor v)
{
    c10::IntArrayRef shape = q.sizes();
    torch::Tensor q_w = q.index({torch::indexing::Slice(), -1});
    torch::Tensor q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    torch::Tensor a = v * (2.0 * torch::pow(q_w, 2) - 1.0).unsqueeze(-1);
    torch::Tensor b = torch::cross(q_vec, v, /*dim=*/-1) * q_w.unsqueeze(-1) * 2.0;
    torch::Tensor c = q_vec * torch::bmm(q_vec.view({shape[0], 1, 3}), v.view({shape[0], 3, 1})).squeeze(-1) * 2.0;
    return a - b + c;
}

void State_RL::_observations_compute()
{
    // 获取当前的观测信息

    torch::Tensor base_quat = torch::tensor({{_lowState->imu.quaternion[1], _lowState->imu.quaternion[2], _lowState->imu.quaternion[3], _lowState->imu.quaternion[0]}});

    // 重力观测
    torch::Tensor projected_gravity = QuatRotateInverse(base_quat, this->_gravity_vec);

    // 命令观测
    _userValue = _lowState->userValue;  // 获取用户输入
    _getUserCmd();  // 解析用户输入命令，更新速度命令
    torch::Tensor commands = torch::cat({torch::tensor({_vCmdBody(0)}), torch::tensor({_vCmdBody(1)}), torch::tensor({_dYawCmd})}, -1);

    // 相对关节角度
    torch::Tensor dof_pos_tensor_ = torch::tensor({_lowState->motorState[3].q-this->_default_dof_pos[3], _lowState->motorState[4].q-this->_default_dof_pos[4], _lowState->motorState[5].q-this->_default_dof_pos[5],
                                                  _lowState->motorState[0].q-this->_default_dof_pos[0], _lowState->motorState[1].q-this->_default_dof_pos[1], _lowState->motorState[2].q-this->_default_dof_pos[2],
                                                  _lowState->motorState[9].q-this->_default_dof_pos[9], _lowState->motorState[10].q-this->_default_dof_pos[10], _lowState->motorState[11].q-this->_default_dof_pos[11],
                                                  _lowState->motorState[6].q-this->_default_dof_pos[6], _lowState->motorState[7].q-this->_default_dof_pos[7], _lowState->motorState[8].q-this->_default_dof_pos[8]});

//    torch::Tensor dof_pos_tensor_ = torch::tensor({_lowState->motorState[3].q, _lowState->motorState[4].q, _lowState->motorState[5].q,
//                                                   _lowState->motorState[0].q, _lowState->motorState[1].q, _lowState->motorState[2].q,
//                                                   _lowState->motorState[9].q, _lowState->motorState[10].q, _lowState->motorState[11].q,
//                                                   _lowState->motorState[6].q, _lowState->motorState[7].q, _lowState->motorState[8].q});

    float dof_pos_temp[12] = {_lowState->motorState[3].q, _lowState->motorState[4].q, _lowState->motorState[5].q,
                              _lowState->motorState[0].q, _lowState->motorState[1].q, _lowState->motorState[2].q,
                              _lowState->motorState[9].q, _lowState->motorState[10].q, _lowState->motorState[11].q,
                              _lowState->motorState[6].q, _lowState->motorState[7].q, _lowState->motorState[8].q};
    saveArrayToFile(dof_pos_temp, 12, "obs_values_cpp.txt");

    torch::Tensor dof_pos_tensor = torch::zeros_like(dof_pos_tensor_);
    for(int i=0; i<12; i++)
    {
        dof_pos_tensor[i] = dof_pos_tensor_[this->dofmapping_obs2lab[i]];
    }

    // 关节角速度
    torch::Tensor dof_vel_tensor_ = torch::tensor({_lowState->motorState[3].dq, _lowState->motorState[4].dq, _lowState->motorState[5].dq,
                                                  _lowState->motorState[0].dq, _lowState->motorState[1].dq, _lowState->motorState[2].dq,
                                                  _lowState->motorState[9].dq, _lowState->motorState[10].dq, _lowState->motorState[11].dq,
                                                  _lowState->motorState[6].dq, _lowState->motorState[7].dq, _lowState->motorState[8].dq});

    torch::Tensor dof_vel_tensor = torch::zeros_like(dof_vel_tensor_);
    for(int i=0; i<12; i++)
    {
        dof_vel_tensor[i] = dof_vel_tensor_[this->dofmapping_obs2lab[i]];
    }

    // 角速度
    torch::Tensor body_ang_vel = torch::tensor({{_lowState->imu.gyroscope[0], _lowState->imu.gyroscope[1], _lowState->imu.gyroscope[2]}});
    //torch::Tensor body_ang_vel = QuatRotateInverse(base_quat, ang_vel);

    // torch::Tensor contact_states = torch::tensor({(*_contact)(1),(*_contact)(0),(*_contact)(3),(*_contact)(2)});

    this->_observation = torch::cat({
        commands.view({1, -1}) * scale_commands.view({1, -1}),
        body_ang_vel.view({1, -1}) * scale_ang_vel,
        projected_gravity.view({1, -1}),
        dof_pos_tensor.view({1, -1}) * scale_dof_pos,
        dof_vel_tensor.view({1, -1}) * scale_dof_vel,
        _action.view({1, -1})}, -1);

}

void State_RL::_getUserCmd(){
    /* Movement */
    _vCmdBody(0) =  invNormalize(_userValue.ly, _vxLim(0), _vxLim(1));
    _vCmdBody(1) = -invNormalize(_userValue.lx, _vyLim(0), _vyLim(1));
    _vCmdBody(2) = 0;

    /* Turning */
    _dYawCmd = -invNormalize(_userValue.rx, _wyawLim(0), _wyawLim(1));
    _dYawCmd = 0.9*_dYawCmdPast + (1-0.9) * _dYawCmd;
    _dYawCmdPast = _dYawCmd;
}

void State_RL::_obs_buffer_update()
{
    _observations_compute(); // 获取新观测
    // this->_obs_buffer_tensor = torch::cat({this->_obs_buffer_tensor.narrow(1, this->_num_obs, (this->_num_obs_history*this->_num_obs-this->_num_obs)), this->_observation.view({1, this->_num_obs})}, -1);
    this->_obs_buffer_tensor = torch::cat(
        {this->_observation.view({1, this->_num_obs}),
        this->_obs_buffer_tensor.narrow(1, 0, (this->_num_obs_history*this->_num_obs-this->_num_obs))}, 
        -1);
}

void State_RL::_action_compute()
{
    // 以50hz调用rl_policy来生成关节角度
    // torch::Tensor latent = _adapt_module.forward({this->_obs_buffer_tensor}).toTensor();
    // this->_action = _body_module.forward({torch::cat({_observation, latent}, -1)}).toTensor();
    long long start_Time = getSystemTime();
    this->_action = _body_module.forward({this->_obs_buffer_tensor}).toTensor();
    long long end_Time = getSystemTime();
    std::cout << "forward time: " << end_Time-start_Time << std::endl;

    this->_action = torch::clamp(this->_action, -clip_actions, clip_actions);

    torch::Tensor actions_scaled = this->_action * this->action_scale;
    int indices[] = {0, 3, 6, 9};
    for (int i : indices)
        actions_scaled[0][i] *= this->hip_scale_reduction; // 模型顺序

    torch::Tensor actions_scaled_ = torch::zeros_like(actions_scaled);

    for(int i=0; i<12; i++)
    {
        actions_scaled_[0][i] = actions_scaled[0][this->dofmapping_lab2obs[i]];
    }

    for(int i=0; i<12; i++)
    {
        this->_targetPos_rl[i] = actions_scaled_[0][this->dof_mapping[i]].item<double>();
        this->_targetPos_rl[i] += this->_default_dof_pos[i];
    }

    saveArrayToFile(this->_targetPos_rl, 12, "action_values_cpp.txt");
}

void State_RL::enter()
{
    std::cout << "[INFO]: Start Joint Position Control learned by RL" << std::endl;
    std::cout << "[INFO]: RL FSM Start Init!" << std::endl;

    for (int i = 0; i < 12; i++)
    {
        _lowCmd->motorCmd[i].mode = 10;
        _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;  // 设置当前关节角度
        _lowCmd->motorCmd[i].dq = 0;
        _lowCmd->motorCmd[i].Kp = this->Kp;
        _lowCmd->motorCmd[i].Kd = this->Kd;
        _lowCmd->motorCmd[i].tau = 0;

        // 设置进入状态机时的初始关节角度为目标角度
        this->_targetPos_rl[i] = this->_default_dof_pos[i];  // 初始pd调整至默认关节位置
        this->_last_targetPos_rl[i] = _lowState->motorState[i].q;
        this->_joint_q[i] = this->_default_dof_pos[i];
    }

    _init_buffers(); // 初始化参数
    _loadPolicy();  // 刚进入RL状态时先加载模型

    // 初始化_obs_buffer_tensor，并获取满足历史时间步长度的观测buffer
    this->_obs_buffer_tensor = torch::zeros({1, this->_num_obs_history*_num_obs});
    _action = torch::zeros({1, 12});
    for (int i = 0; i < this->_num_obs_history; ++i) {
        _obs_buffer_update(); // 更新初始的观测buffer
    }
}

void State_RL::step()
{
    // send control signal to actuator
    for (int j = 0; j < 12; j++)
    {
        _lowCmd->motorCmd[j].mode = 10;
        _lowCmd->motorCmd[j].q = this->_targetPos_rl[j];
        _lowCmd->motorCmd[j].dq = 0;
        _lowCmd->motorCmd[j].Kp = this->Kp;
        _lowCmd->motorCmd[j].Kd = this->Kd;
        _lowCmd->motorCmd[j].tau = 0;
    }
}

void State_RL::run()
{
    // 200HZ
    ++this->count;
    if (this->count > this->decimation)
    {
        // 50HZ
        _obs_buffer_update();
        _action_compute();
        this->count = 0;
    }
    step();
}

void State_RL::exit()
{
    this->_percent_1 = 0;
}

FSMStateName State_RL::checkChange()
{
    if (_lowState->userCmd == UserCommand::L2_B)
    {
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::L2_A){
        return FSMStateName::FIXEDSTAND;
    }
    else{
        return FSMStateName::RL;
    }
}