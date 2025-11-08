## File structure:
- a_fixed_parameters_to_ini.py: takes the fixed parameters (= same for every point in the chain, e.g. do_shooting) and writes a temporary ini file  
- b_generate_ini_from_chain.py: takes a random point in the given chain and writes an ini file (ready for class) 
- c_ini_to_python.py turns the .ini into a python dict (ready for classy)  
- d_run_class.py runs classy, computes background quantities (f_ede_mscf) 
e_plot.py plots the desired background quantities 

main.py runs a loop on the number of samples we wish to take, which chain, etc. (contains most config info)


