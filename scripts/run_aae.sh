CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name="aae" \
    --aae.prompt="a cat and a dog" \
    --aae.token_indices=[2,5] \
    --aae.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=1 python main.py \
    --exp_name="aae" \
    --aae.prompt="a apple and a frog" \
    --aae.token_indices=[2,5] \
    --aae.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=2 python main.py \
    --exp_name="aae" \
    --aae.prompt="a car and a sheep" \
    --aae.token_indices=[2,5] \
    --aae.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=3 python main.py \
    --exp_name="aae" \
    --aae.prompt="a bench and a boat" \
    --aae.token_indices=[2,5] \
    --aae.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=4 python main.py \
    --exp_name=aae \
    --aae.prompt="a bird and a clock" \
    --aae.token_indices=[2,5] \
    --aae.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=5 python main.py \
    --exp_name="aae" \
    --aae.prompt="a cup and a vase" \
    --aae.token_indices=[2,5] \
    --aae.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=6 python main.py \
    --exp_name="aae" \
    --aae.prompt="a vase and a book" \
    --aae.token_indices=[2,5] \
    --aae.seeds=[42,1989,293850] &

wait

CUDA_VISIBLE_DEVICES=6 python main.py \
    --exp_name="aae" \
    --aae.prompt="a red car and a white sheep" \
    --aae.token_indices=[2,3,6,7] \
    --aae.seeds=[42,1989,293850] &

wait