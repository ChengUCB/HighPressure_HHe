###############################################################################
# MD simulation for PBE C/H with NN potential
###############################################################################

###############################################################################
# VARIABLES
###############################################################################
variable runnerCutoff     equal  4.24                                            # largest symmetry function cutoff (Angstrom)
#variable runnerDir       string "../../../nnp"	                        # directory containing RuNNer files
###############################################################################
# SETUP
###############################################################################
#
# set up pair style runner
#pair_style nnp dir ${runnerDir} showew no showewsum 1000 resetew yes maxew 20000 cflength 1.8897261328 cfenergy 0.0367493254
# set up pair style coefficients
#pair_coeff * * ${runnerCutoff}


variable nnDir1    string "../../../../nnp-r-4-gen6-PBE/"
variable nnDir2    string "../../../../nnp-r-1-gen5-DELTA/"
pair_style hybrid/overlay nnp dir ${nnDir1} showew no showewsum 10 resetew yes maxew 100000 cflength 1.8897261328 cfenergy 0.0367493254 nnp dir ${nnDir2} showew no showewsum 20 resetew yes maxew 100000 cflength 1.8897261328 cfenergy 0.0367493254
pair_coeff * * nnp 1 ${runnerCutoff}        # set up pair style coefficients
pair_coeff * * nnp 2 ${runnerCutoff}        # set up pair style coefficients
