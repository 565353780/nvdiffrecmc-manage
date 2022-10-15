#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse


def getFLAGS():
    parser = argparse.ArgumentParser(description='nvdiffrecmc')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr',
                        '--texture-res',
                        nargs=2,
                        type=int,
                        default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mip',
                        '--custom-mip',
                        action='store_true',
                        default=False)
    parser.add_argument('-bg',
                        '--background',
                        default='checker',
                        choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument(
        '--loss',
        default='logl1',
        choices=['logl1', 'logl2', 'mse', 'smape', 'relativel2'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    # Render specific arguments
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--bsdf',
                        type=str,
                        default='pbr',
                        choices=['pbr', 'diffuse', 'white'])
    # Denoiser specific arguments
    parser.add_argument('--denoiser',
                        default='bilateral',
                        choices=['none', 'bilateral'])
    parser.add_argument('--denoiser_demodulate', type=bool, default=True)

    FLAGS = parser.parse_args()

    FLAGS.mtl_override = None  # Override material of model
    FLAGS.dmtet_grid = 64  # Resolution of initial tet grid. We provide 64 and 128 resolution grids.
    #    Other resolutions can be generated with https://github.com/crawforddoran/quartet
    #    We include examples in data/tets/generate_tets.py
    FLAGS.mesh_scale = 2.1  # Scale of tet grid box. Adjust to cover the model
    FLAGS.envlight = None  # HDR environment probe
    FLAGS.env_scale = 1.0  # Env map intensity multiplier
    FLAGS.probe_res = 256  # Env map probe resolution
    FLAGS.learn_lighting = True  # Enable optimization of env lighting
    FLAGS.display = None  # Configure validation window/display. E.g. [{"bsdf" : "kd"}, {"bsdf" : "ks"}]
    FLAGS.transparency = False  # Enabled transparency through depth peeling
    FLAGS.lock_light = False  # Disable light optimization in the second pass
    FLAGS.lock_pos = False  # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer = 0.2  # Weight for sdf regularizer.
    FLAGS.laplace = "relative"  # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale = 3000.0  # Weight for Laplace regularizer. Default is relative with large weight
    FLAGS.pre_load = True  # Pre-load entire dataset into memory for faster training
    FLAGS.no_perturbed_nrm = False  # Disable normal map
    FLAGS.decorrelated = False  # Use decorrelated sampling in forward and backward passes
    FLAGS.kd_min = [0.0, 0.0, 0.0, 0.0]
    FLAGS.kd_max = [1.0, 1.0, 1.0, 1.0]
    FLAGS.ks_min = [0.0, 0.08, 0.0]
    FLAGS.ks_max = [0.0, 1.0, 1.0]
    FLAGS.nrm_min = [-1.0, -1.0, 0.0]
    FLAGS.nrm_max = [1.0, 1.0, 1.0]
    FLAGS.clip_max_norm = 0.0
    FLAGS.cam_near_far = [0.1, 1000.0]
    FLAGS.lambda_kd = 0.1
    FLAGS.lambda_ks = 0.05
    FLAGS.lambda_nrm = 0.025
    FLAGS.lambda_nrm2 = 0.25
    FLAGS.lambda_chroma = 0.0
    FLAGS.lambda_diffuse = 0.15
    FLAGS.lambda_specular = 0.0025
    return FLAGS
