# mnist denoising experiments
#experiments with changing num_imgs
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 5 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 75 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 150 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 200 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1

#experiments with augmented images, changing num_imgs
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 5 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 75 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 150 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 200 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --rand_shift --batch_size 1

#experiments with changing latent_dim
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 20 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 80 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 150 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_epochs 10000 --batch_size 1

#experiments with changing num_samples
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_samples 8 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_samples 25 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_samples 50 --num_epochs 10000 --batch_size 1
#python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --num_samples 150 --num_epochs 10000 --batch_size 1

#experiments with changing learning rate
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --lr 1e-6 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --lr 1e-5 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --lr 1e-3 --num_epochs 10000 --batch_size 1
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm --normalize_loss --lr 1e-2 --num_epochs 10000 --batch_size 1
