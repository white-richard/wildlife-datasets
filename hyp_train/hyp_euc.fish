sudo docker container stop optuna-pg
sudo docker container rm optuna-pg
sudo docker volume rm optuna_pgdata
sudo docker run -d --name optuna-pg \
  -e POSTGRES_USER=optuna \
  -e POSTGRES_PASSWORD=optuna-pg \
  -e POSTGRES_DB=wr10k \
  -v optuna_pgdata:/var/lib/postgresql/data \
  -p 100.121.43.41:5432:5432 \
  postgres:15

python train_optuna_sweep_distill.py \
  --tune-trials 20 \
  --tune-direction maximize \
  --tune-storage postgresql+psycopg2://optuna:optuna-pg@100.121.43.41:5432/wr10k \
  --tune-study hyp_vs_euc \
  --wandb online --project hyp_vs_euc_sweep_size \
  --tune-seed 42 \
  --hyperbolic \
  --model-name "hyp_swin_fix" \
  --tune-min-epochs 50 \
  --output-embed-dim 64

# sudo docker container stop optuna-pg
# sudo docker container rm optuna-pg
# sudo docker volume rm optuna_pgdata
# sudo docker run -d --name optuna-pg \
#   -e POSTGRES_USER=optuna \
#   -e POSTGRES_PASSWORD=optuna-pg \
#   -e POSTGRES_DB=wr10k \
#   -v optuna_pgdata:/var/lib/postgresql/data \
#   -p 100.121.43.41:5432:5432 \
#   postgres:15

# python train_optuna_sweep_distill.py \
#   --tune-trials 20 \
#   --tune-direction maximize \
#   --tune-storage postgresql+psycopg2://optuna:optuna-pg@100.121.43.41:5432/wr10k \
#   --tune-study hyp_vs_euc \
#   --wandb online --project hyp_vs_euc_sweep_size \
#   --model-name "ViT" \
#   --tune-seed 42 \
#   --tune-min-epochs 50 \
#   --output-embed-dim 64
