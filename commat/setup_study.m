% setup_study.m
function model = setup_study(model)
    comp = model.component('comp1');
    mesh = comp.mesh.create('mesh1');
    mesh.feature('size').set('hmax', 1.5);
    mesh.run;
    std = model.study.create('std1');
    para = std.feature.create('p1', 'Parametric');
    para.set('pname', 'para_ramp');
    para.set('plistarr', '0.1, 0.2, 0.4, 0.6, 0.8, 1.0');
    stat = std.create('stat', 'Stationary');
    sol = model.sol.create('sol1');
    sol.study('std1');
    sol.create('st1', 'StudyStep');
    sol.feature('st1').set('study', 'std1');
    sol.feature('st1').set('studystep', 'stat');
    sol.create('v1', 'Variables');
    sol.create('s1', 'Stationary');
    s1 = sol.feature('s1');
    s1.feature.remove('seg1');
    fc = s1.feature.create('fc1', 'FullyCoupled');
    fc.set('maxiter', 50);
end
