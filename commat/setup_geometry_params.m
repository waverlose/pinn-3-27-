% setup_geometry_params.m
function model = setup_geometry_params()
    import com.comsol.model.*
    import com.comsol.model.util.*
    
    model = ModelUtil.create('IVD_PINN_Benchmark');
    comp = model.component.create('comp1', true);
    
    %% 1. Parameters (Mirroring config.py)
    p = model.param;
    p.set('GEO_A', '20.0'); p.set('GEO_B', '15.0'); p.set('GEO_H', '10.0');
    p.set('NP_RATIO', '0.7');
    
    p.set('MAT_NP_MU', '0.06'); p.set('MAT_AF_MU', '0.14');
    p.set('MAT_NP_LAMBDA', '0.04'); p.set('MAT_AF_LAMBDA', '0.17');
    p.set('MAT_NP_WSR', '0.15'); p.set('MAT_AF_WSR', '0.30');
    p.set('MAT_FIBER_ALPHA_MAX', '5.0');
    p.set('MAT_FIBER_THETA_INNER', '30.0');
    
    p.set('PHY_R', '8314.0'); p.set('PHY_T', '310.0'); p.set('PHY_C_EXT', '1.5e-7');
    p.set('VISCO_SI', '1e-3');
    p.set('ION_RP', '0.2e-6'); p.set('ION_DP0', '1.33e-3');
    
    p.set('MAT_NP_PERM_A', '2.48e-5'); p.set('MAT_NP_PERM_N', '2.15');
    p.set('MAT_AF_PERM_A', '3.10e-5'); p.set('MAT_AF_PERM_N', '2.20');
    p.set('MAT_NP_DIFF_A', '1.25'); p.set('MAT_NP_DIFF_B', '0.68');
    p.set('MAT_AF_DIFF_A', '1.29'); p.set('MAT_AF_DIFF_B', '0.37');
    
    p.set('BC_TOP_PRESSURE', '0.3');
    p.set('para_ramp', '0');

    %% 2. Geometry
    geom = comp.geom.create('geom1', 3);
    wp = geom.create('wp1', 'WorkPlane');
    
    % Domain partitioning
    c1 = wp.geom.create('c1', 'Circle'); c1.set('r', 'GEO_A');
    sca1 = wp.geom.create('sca1', 'Scale'); sca1.selection('input').set({'c1'});
    sca1.set('type', 'anisotropic'); sca1.set('factor', {'1', 'GEO_B/GEO_A', '1'});
    
    c2 = wp.geom.create('c2', 'Circle'); c2.set('r', 'GEO_A*NP_RATIO');
    sca2 = wp.geom.create('sca2', 'Scale'); sca2.selection('input').set({'c2'});
    sca2.set('type', 'anisotropic'); sca2.set('factor', {'1', 'GEO_B/GEO_A', '1'});
    
    ext = geom.create('ext1', 'Extrude');
    ext.selection('input').set({'wp1'});
    ext.set('distance', 'GEO_H');
    geom.run;
    
    %% 3. Selections (Precise Boundary Mapping)
    % Domains
    sel_np = comp.selection.create('sel_np', 'Cylinder');
    sel_np.set('entitydim', 3); sel_np.set('r', 'GEO_A*NP_RATIO + 0.1');
    
    % Surface: Bottom (z=0)
    sel_btm = comp.selection.create('sel_btm', 'Box');
    sel_btm.set('entitydim', 2); sel_btm.set('zmin', -0.1); sel_btm.set('zmax', 0.1);
    
    % Surface: Top (z=H)
    sel_top = comp.selection.create('sel_top', 'Box');
    sel_top.set('entitydim', 2); sel_top.set('zmin', 'GEO_H-0.1'); sel_top.set('zmax', 'GEO_H+0.1');
    
    % Surface: Side (r=Outer)
    sel_side = comp.selection.create('sel_side', 'Cylinder');
    sel_side.set('entitydim', 2); sel_side.set('r', 'GEO_A-0.1'); sel_side.set('condition', 'outside');
    
    % Surface: NP Top part
    sel_np_top = comp.selection.create('sel_np_top', 'Cylinder');
    sel_np_top.set('entitydim', 2); sel_np_top.set('r', 'GEO_A*NP_RATIO + 0.1');
    sel_np_top.set('zmin', 'GEO_H-0.1'); sel_np_top.set('zmax', 'GEO_H+0.1');
    
    % Surface: NP Bottom part
    sel_np_btm = comp.selection.create('sel_np_btm', 'Cylinder');
    sel_np_btm.set('entitydim', 2); sel_np_btm.set('r', 'GEO_A*NP_RATIO + 0.1');
    sel_np_btm.set('zmin', -0.1); sel_np_btm.set('zmax', 0.1);
    
    % Chemistry Dirichlet Selection: NP_Top + NP_Btm + Outer_Side
    sel_chem_dir = comp.selection.create('sel_chem_dir', 'Union');
    sel_chem_dir.set('entitydim', 2);
    sel_chem_dir.selection('input').set({'sel_np_top', 'sel_np_btm', 'sel_side'});

    fprintf('Geometry and precise selections setup successfully.\n');
end
