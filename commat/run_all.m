% run_all.m
clear; clc;
addpath(genpath(pwd));
fprintf('Building PINN-Aligned Weak Form Model...\n');
model = setup_geometry_params();
model = setup_variables(model);
model = setup_weak_physics(model);
model = setup_study(model);
fprintf('Build complete. Run model.sol(''sol1'').runAll to compute.\n');
