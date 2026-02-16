python main.py --output_dir ./results -c config/cfg_surgcount_vit_b.py --datasets config/datasets_surgcount.json --pretrain_model_path ./results/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased --eval --save_results
python visualize.py
