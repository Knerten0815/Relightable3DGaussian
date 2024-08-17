@echo off
set root_dir=datasets/nerf_synthetic/
set list=ficus

for %%i in (%list%) do (
    python train.py --eval -s %root_dir%%%i/ -m output/NeRF_Syn/%%i/neilf -c output/NeRF_Syn/%%i/3dgs/chkpnt30000.pth --save_training_vis --position_lr_init 0.000016 --position_lr_final 0.00000016 --normal_lr 0.001 --sh_lr 0.00025 --opacity_lr 0.005 --scaling_lr 0.0005 --rotation_lr 0.0001 --iterations 40000 --lambda_base_color_smooth 0 --lambda_roughness_smooth 0 --lambda_light_smooth 0 --lambda_light 0.01 -t neilf --sample_num 64 --save_training_vis_iteration 200 --lambda_env_smooth 0.01
    
    python eval_nvs.py --eval -m output/NeRF_Syn/%%i/neilf -c output/NeRF_Syn/%%i/neilf/chkpnt40000.pth -t neilf
)