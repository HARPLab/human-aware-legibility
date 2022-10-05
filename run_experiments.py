import pipeline_generate_paths as pipeline
import utility_legibility as legib

def exp_diff_legibilities():
	legib_options = legib.get_legibility_options()

	for l in legib_options:
		exp_settings 							= pipeline.get_default_exp_settings()
		exp_settings[pipeline.SETTING_LEGIBILITY_METHOD] = l

		print("Testing label " + l)
		pipeline.do_exp(exp_settings)
		print("Done")
		print("~~~~~~~~~~~")
	
	print("Done with experiments of diff legibilities")


def exp_determine_lam_eps():
	lam_vals = []
	eps = 1e-7
	# eps_vals = []
	# # exit()
	# * 1e-6
	for i in range(-5, -10, -1):
	# for i in np.arange(1.1, 2, .1):
		new_val = 10 ** i 
	# 	eps_vals.append(new_val)
		lam_vals.append(new_val)
	# 	# lam_vals.append(new_val)

	# print("REMIX TIME")
	# for eps in eps_vals:
	lam = 0
	angle_strs = [520]
	rbs = [55]
	
	print("WILDIN")
	for astr in angle_strs:
		for rb in rbs:
			exp_settings = pipeline.get_default_exp_settings()
			exp_settings[pipeline.SETTING_LAMBDA] = lam
			exp_settings[pipeline.SETTING_ANGLE_STRENGTH] = astr
			exp_settings[pipeline.SETTING_RIGHT_BOUND] = rb

			do_exp(exp_settings)
	
def exp_observer_aware():
	lam = 0
	kill_mode = True
	astr = 500
	rb = 40

	# print("Doing main")
	# Get the best path for the given scenario
	exp_settings 							= pipeline.get_default_exp_settings('jul15')
	exp_settings[pipeline.SETTING_LAMBDA] 			= lam
	exp_settings[pipeline.SETTING_ANGLE_STRENGTH] 	= astr
	exp_settings[pipeline.SETTING_RIGHT_BOUND] 		= rb

	do_exp(exp_settings)

def main():
	# export_best_options()
	# exit()

	exp_diff_legibilities()


if __name__ == "__main__":
	main()