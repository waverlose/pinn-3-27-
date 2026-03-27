% setup_weak_physics.m
function model = setup_weak_physics(model)
    comp = model.component('comp1');
    w_solid = comp.physics.create('w_solid', 'WeakFormPDE', 'geom1');
    w_solid.field('dimensionless').component({'u', 'v', 'w'});
    weak_solid_expr = ['(P11*test(uX) + P12*test(uY) + P13*test(uZ) + ', ...
                       ' P21*test(vX) + P22*test(vY) + P23*test(vZ) + ', ...
                       ' P31*test(wX) + P32*test(wY) + P33*test(wZ)) + ', ...
                       ' (BC_TOP_PRESSURE * para_ramp * test(w))'];
    w_solid.feature('wfeq1').set('weak', weak_solid_expr);
    d_btm = w_solid.create('dir_btm', 'DirichletBoundary', 2);
    d_btm.selection.named('sel_btm');
    d_btm.set('component', {'u', 'v', 'w'});
    d_btm.set('r', {'0', '0', '0'});
    d_top = w_solid.create('dir_top', 'DirichletBoundary', 2);
    d_top.selection.named('sel_top');
    d_top.set('component', {'u', 'v'});
    d_top.set('r', {'0', '0'});
    w_fluid = comp.physics.create('w_fluid', 'WeakFormPDE', 'geom1');
    w_fluid.field('dimensionless').component({'mu_w'});
    w_fluid.feature('wfeq1').set('weak', 'JwX*test(mu_wX) + JwY*test(mu_wY) + JwZ*test(mu_wZ)');
    d_f = w_fluid.create('dir_f', 'DirichletBoundary', 2);
    d_f.selection.named('sel_chem_dir');
    d_f.set('r', '0');
    w_ion = comp.physics.create('w_ion', 'WeakFormPDE', 'geom1');
    w_ion.field('dimensionless').component({'mu_i'});
    w_ion.feature('wfeq1').set('weak', 'JpX*test(mu_iX) + JpY*test(mu_iY) + JpZ*test(mu_iZ)');
    d_i = w_ion.create('dir_i', 'DirichletBoundary', 2);
    d_i.selection.named('sel_chem_dir');
    d_i.set('r', '0');
end
