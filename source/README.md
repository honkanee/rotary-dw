# ROTARY-DW
...

## Relaxation

DW is relaxed in Bext=0 for nsteps_relax iterations.

## Ramped field

Ramped version is applied if all of these are true 

    - ramped_field is set to true 
    - Bext for the first and last simulations are on the ascending order when comparing their absolute values 
    - Bext step is nonzero 
    - Adding step to B_ext makes field closer to value of Bext_end instead od diverging 


If any of the conditions above is false, regular simulation is run

    - Bext_step, Bext_end, nsteps_at_ramp values have no impact on the simulation 
    - Simulation is run with given B_ext for nsteps iterations 


In ramped version

    - Simulation is run for nsteps_at_ramp iterations at each Bext value of B_ext=n*Bext_step, where n=0,1,... until Bext_end is achieved 
    - nsteps has no impact on this version 