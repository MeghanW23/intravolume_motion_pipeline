%%% Arvind Balachandrasekaran
% Code to simultaneous motion compensated reconstruction and slice time
% correction in rsfMRI

% data_path = '/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/p004-ses-04_outputs/';
% fmri_fname = '/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/participant_data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.nii.gz';
% fmri_fname_bgremoved = '/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/p004-ses-04_outputs/p004-ses-04_func-bold_task-NFB2_20250827181314_24_bgremoved.nii.gz';
% OutputMotionParams = '/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/p004-ses-04_outputs/parameters.txt';
% abruptmotion_fname  = '/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/p004-ses-04_outputs/recon_abruptmotion.nii.gz';
% sms_fac=4; 

function main = main_cameraparams(data_path, fmri_fname, fmri_fname_bgremoved, OutputMotionParams, abruptmotion_fname, main_script_dir, directliftandunliftcodes_dir, operators_dir, sms_fac, thresh_forscrub, slice_timing)

	% Add Python module directory to MATLAB Python path
	% Add main script directory to Python path
	if ~any(strcmp(cell(py.sys.path), main_script_dir))
		insert(py.sys.path, int32(0), main_script_dir);
	end

	% Add operators directory too (if needed)
	if ~any(strcmp(cell(py.sys.path), operators_dir))
		insert(py.sys.path, int32(0), operators_dir);
	end

	py.importlib.import_module("pyfuncs_forsltovol")

	disp("----------------------------------------------------------------")
	disp("------ main_cameraparams.m received the following inputs: ------")
	disp(" ")
	disp("Output Directory:")
	disp(data_path)
	disp(" ")
	disp("Input NiFTI Image:")
	disp(fmri_fname)
	disp(" ")
	disp("Background Removed Input NiFTI Image:")
	disp(fmri_fname_bgremoved)
	disp(" ")
	disp("Motion Parameters Textfile:")
	disp(OutputMotionParams)
	disp(" ")
	disp("Abrupt Motion Output Filename:")
	disp(abruptmotion_fname)
	disp(" ")
	disp("Main Motion Reconstruction Script Directory:")
	disp(main_script_dir)
	disp(" ")
	disp("directliftandunliftcodes Directory:")
	disp(directliftandunliftcodes_dir)
	disp(" ")
	disp("Operators Directory:")
	disp(operators_dir)
	disp(" ")
	disp("SMS Factor:")
	disp(sms_fac)
	disp(" ")
	disp("Scrubbing Threshold:")
	disp(thresh_forscrub)
	disp(" ")
	disp("Slice Timing:")
	disp(slice_timing)
	disp("----------------------------------------------------------------")
	disp("----------------------------------------------------------------")

	addpath(main_script_dir);
	addpath(directliftandunliftcodes_dir);
	addpath(operators_dir);

	% if multiple slices are acquired at the same time, sms > 1
	sms_fac=str2double(sms_fac); 

	% thresh_noscrub = 0; % variable indicating no scrubbing
	thresh_forscrub = str2double(thresh_forscrub); % variable indicating volumes with framewise displacement > 0.6 will be marked for scrubbing

	% convert input slice_timing string to row vector
	slice_timing = str2num(slice_timing);

	[~,f,~] = fileparts(fmri_fname);
	[~,g,~] = fileparts(f);

	Y = py.SimpleITK.ReadImage(fmri_fname,py.SimpleITK.sitkFloat64);
	origin_4d = Y.GetOrigin();
	spacing_4d = Y.GetSpacing();
	direction_4d=  Y.GetDirection();
	%Iorig = numpytomatlab(py.SimpleITK.GetArrayFromImage(Y));
	Iorig =  permute(double(py.SimpleITK.GetArrayFromImage(Y)),[3,4,2,1]);
	[n1,n2,nsl,nvs] = size(Iorig);
	opt.vol_start = 1; % Adjust this parameter to discard initial frames, vol_start =  No of FramesTodiscard + 1
	nv = nvs-opt.vol_start+1;
	if opt.vol_start ~=1
		Iorig_tr = Iorig(:,:,:,opt.vol_start:nvs);
		Iorig_tr_np = py.numpy.asarray(permute(Iorig_tr,[4,3,1,2]));
		Y = py.pyfuncs_forsltovol.numpy4Dtositk(Iorig_tr_np,origin_4d,direction_4d,spacing_4d,int32(nv));
	end
	opt.I1 = Iorig(:,:,:,opt.vol_start);

	opt.nv = nv;
	nslbysmsfac = nsl/sms_fac;
	opt.nslbysmsfac = nslbysmsfac;
	opt.sms_fac = sms_fac;

	% Get slice acquisition order
	[~, opt.slice_acq_order] = sort(slice_timing);
	opt.params_ind = py.numpy.int32(py.range(int32(nslbysmsfac)));
	%const = 2*ones(1,nslbysmsfac);

	nGroups = nsl / sms_fac
	const = sms_fac * ones(1, nGroups);
	sl_acq_order_cell_py = mat2cell(opt.slice_acq_order-1,[1],const); % python convention.

	%ilacq_cell_mat = mat2cell(opt.slice_acq_order,[1],const);  % matlab convention.
	slice_info = py.dict(pyargs());
	for i = 1:nslbysmsfac
		slice_info{i-1} =  matlabtonumpy(int32(sl_acq_order_cell_py{i}));% according to python convention.
	end
	opt.slice_info = slice_info;
	%%
	% Estimate motion parameters
	%OutputNiftiFile = strcat(data_path,'motionparams/',g,'_vvr.nii.gz');
	%OutputMotionParams = strcat(data_path,'cameraparams_abruptnodding.txt');

	%%%% Here SLice to registration algorithm that estimates slice level motion
	%%%% estimates should be used.
	% if ~exist(OutputMotionParams)
	%     EstimateMotionParamsFromMotionData(fmri_fname,opt.vol_start,OutputNiftiFile,OutputMotionParams);
	% end
	%% Compute framewse displacement
	params_temp = load(OutputMotionParams);
	[r,c] = size(params_temp);
	params_temp = params_temp(opt.vol_start:r,:);
	%params = matlabtonumpy(params);
	opt.params = py.numpy.asarray(params_temp);   %%% numpy stored in opt

	params_temp(:,1:3) = params_temp(:,1:3)*50;%% displacement = r times theta, where r = 35 mm for DHCP
	params_diff = params_temp(1:end-1,:) - params_temp(2:end,:);
	SWD = sum(abs(params_diff),2); % frame wise displacement
	SWD = [0;SWD];
	%figure(1),plot(SWD);drawnow;
	y = thresh_forscrub*ones(1,length(SWD));
	%hold on,plot(y);drawnow;
	%%
	opt.n1 = n1;
	opt.n2 = n2;
	opt.nsl = nsl;
	%% fmri time series without background
	%fmri_fname_bgremoved = strcat(data_path,'_bgremoved.nii.gz');
	Y_nobg_img = py.SimpleITK.ReadImage(fmri_fname_bgremoved,py.SimpleITK.sitkFloat64);
	I_nobg = permute(double(py.SimpleITK.GetArrayFromImage(Y_nobg_img)),[3,4,2,1]);
	%I_nobg = numpytomatlab(py.SimpleITK.GetArrayFromImage(Ynobg_img));
	I_nobg = I_nobg(:,:,:,opt.vol_start:nvs);
	X_nobg = reshape(I_nobg,[n1*n2*nsl,nv]);
	opt.ind_bg = ~any(X_nobg,2); % index value of 1 corresponds to background pixel.
	clear I_nobg X_nobg Y_nobg_img
	%% set parameters
	opt.mu = 25; % regularization parameter -  From my experiments, I have observed the algorithm to be robust with changing mu.
	%% For now, don't change the value of the pair (beta, beta_fac). Generate the results and we can decide if we want to try a different pair.
	opt.beta = 7.5;%0.25;%7.5;% initial value of beta
	opt.beta_fac = 1.1;%1.2; %1.1;% variable to increment beta every iteration
	%%
	opt.ft = 50;%75;% filter size. Determines the size of Hankel matrix formed at every voxel.
	opt.overall_maxIter = 20;%15; % Maximum number of iterations for the algorithm
	opt.maxIter = 6;% Maximum number of iterations for the x-subproblem
	opt.Njobs = -1;% Uses all cores when the python codes corresponding to the z-subproblem are executed
	opt.timepoint1 = 116819;% For display purposes. time series corresponding to it is displayed
	opt.timepoint2 = 104879;% For display purposes. time series corresponding to it is displayed
	opt.ts_start  = 1; % Starting point for downsampling the reconstructed time series
	opt.tolerance = 1e-4; % Stopping point for X subproblem
	opt.eta = 1.1;% parameter for IRLS algorithm. Usually, eta > 1 and < 1.5opt.I1 = Iorig(:,:,:,opt.vol_start);  % for initialization purposes.
	opt.p = 0.1;% Schatten p-norm value 0 <= p <=1


	[~,X_img,opt] = reconstruct_timeseries_noscrub_forsltovol(Y,opt);


	%fname = strcat(data_path,'recon_',g,'.nii.gz');
	%fname = strcat(data_path,'recon_',g,'_beta0.2_fac1.2_iter20_segnooverlap.nii.gz');
	py.SimpleITK.WriteImage(X_img, abruptmotion_fname);  
	%fname = strcat(data_path,'recon_noscrub_betai nit',num2str(opt.beta),'_',g,'.nii.gz');
		%py.SimpleITK.WriteImage(X_img, fname);
	%X_img = py.SimpleITK.ReadImage(fname,py.SimpleITK.sitkFloat64);
	%% The downsampled output is the input to the second stage, where volumes with FD > threshold are marked for scrubbing.
	%% The missing data is then interpolated using the matrix completion algorithm.
	%% Interpolate scrubbed data.
	opt.lower_thresh = -1;
	opt.higher_thresh = 2; %% Threshold to remove volumes.
	opt.tolerance = 1e-6;
	opt.maxIter = 45;
	opt.thresh_forscrub = thresh_forscrub;
	opt.SWD = SWD;
	opt.beta = 1; %% Don't change.

	%figure(i),plot(SWD);drawnow;
	y = thresh_forscrub*ones(1,length(SWD));
	%hold on,plot(y);drawnow;

	[vol_i,~,~] = find(SWD>=thresh_forscrub);

	if ~isempty(vol_i)
		disp("Volume exceeded the scrubbing threshold. Scrubbing data")
		opt.filt_siz = nv/2;   %% Have set the filter size to length of series/2. If the results are not good, run the algorithm for a different value.
		for count = 1:length(opt.filt_siz)
			% opt.ft = opt.filt_siz(count);
			opt.ft = round(opt.filt_siz(count));
				if opt.ft < 1
					error('Invalid filter size ft = %d', opt.ft);
				end
			[Xs,vol_rem] = MainFunctionForInterpScrubbedData(X_img,opt);
			Is = reshape(Xs,n1,n2,nsl,nv);
			X_np = matlabtonumpy(Is);
			X_img = py.pyfuncs_sv_parallelized_ss_volumelevel.numpy4Dtositk(X_np,origin_4d,direction_4d,spacing_4d,int32(nv));
			filename   = [data_path,'recon_scrub_betainit',num2str(opt.beta),'_','ft' num2str(opt.ft),'_',g,'.nii.gz'] ;
			
			py.SimpleITK.WriteImage(X_img, filename);
			disp("Scrubbed Data at:")
			disp(filename)
		end
		py.numpy.save(strcat(data_path,'vol_rem_thr0.5_afterscrubbing'),matlabtonumpy(vol_rem));
	else
		disp("No volumes exceeded the scrubbing threshold. Will not scrub data. ")
	end
	disp("Intra-Volume Motion Correction Script is Done.")
end