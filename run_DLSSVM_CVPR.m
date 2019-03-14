%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Code of DLSSVM tracker described in
%	"Object Tracking via Dual Linear Structured SVM and Explicit Feature Map", 
%   Jifeng Ning, Jimei Yang, Shaojie Jiang, Lei Zhang and Ming-Hsuan Yang, 
%   CVPR,Las Vegas, 4266-4274, 2016.
%
%	e-mail:shaojiejiang@126.com,jf ning@sina.com, 
%
% ********************************************************************%%%%%%
% you need configure the opencv for run this program.                
% The program is successfully run under opencv 2.4.8                 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function results=run_DLSSVM_CVPR(seq, res_path, bSaveImage)
% function: the interface to the OTB50 and OTB100 proposed by Wu et al CVPR
% 2013 and PAMI15 respectively
results=tracker(seq.path, seq.ext, false, seq.init_rect, seq.startFrame, seq.endFrame, seq.s_frames);

disp(['fps: ' num2str(results.fps)])
results.type = 'rect';
end