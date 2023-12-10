# mnist denoising experiments
#experiments with changing num_imgs
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 5 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 75 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 150 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 200 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1

#experiments with augmented images, changing num_imgs
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 5 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 75 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 150 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 200 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1

#experiments with changing latent_dim
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 20 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 80 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 150 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1

#experiments with changing num_samples
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_samples 8 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_samples 25 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_samples 50 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_samples 150 --num_epochs 10000 --batch_size 1

#experiments with changing learning rate
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --lr 1e-6 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --lr 1e-5 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --lr 1e-3 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --lr 1e-2 --num_epochs 10000 --batch_size 1

#Additional tests

#check no_entropy
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --no_entropy

#changing dropout_val
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --dropout_val 1e-2
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --dropout_val 1e-4
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --dropout_val 1e-6

#changing GMM_EPS
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-2 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-6 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-8 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1

#changing layer_size
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --layer_size 100
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --layer_size 150
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --layer_size 200

#changing num_layer_decoder
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --num_layer_decoder 6
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --num_layer_decoder 12
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1 --num_layer_decoder 24

#changing sigma vals
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.2 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.7 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 1.1 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1

#changing class 
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 4 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 1 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 7 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1

#changing image size
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 28 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 32 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 128 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
