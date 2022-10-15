#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../nvdiffrecmc")

import os
import time
import torch
import numpy as np
import nvdiffrast.torch as dr
from render import obj, material, util, light
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh
from train import createLoss, prepare_batch, validate_itr, initial_guess_material, xatlas_uvmap
from denoiser.denoiser import BilateralDenoiser

from nvdiffrecmc_manage.Config.config import getFLAGS

from nvdiffrecmc_manage.Dataset.nerf import DatasetNERF


def optimize_mesh(denoiser,
                  glctx,
                  glctx_display,
                  geometry,
                  opt_material,
                  lgt,
                  dataset_train,
                  dataset_validate,
                  FLAGS,
                  warmup_iter=0,
                  log_interval=10,
                  pass_idx=0,
                  pass_name="",
                  optimize_light=True,
                  optimize_geometry=True):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(
        FLAGS.learning_rate, list) or isinstance(
            FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(
        learning_rate, list) or isinstance(learning_rate,
                                           tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(
        learning_rate, list) or isinstance(learning_rate,
                                           tuple) else learning_rate
    learning_rate_lgt = learning_rate[2] if isinstance(
        learning_rate, list) or isinstance(learning_rate,
                                           tuple) else learning_rate * 3.0

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter
        return max(
            0.0,
            10**(-(iter - warmup_iter) *
                 0.0002))  # Exponential falloff from [1.0, 0.1] over 5k epochs

    trainable_list = material.get_parameters(opt_material)

    if optimize_light:
        optimizer_light = torch.optim.Adam(
            (lgt.parameters() if lgt is not None else []),
            lr=learning_rate_lgt)
        scheduler_light = torch.optim.lr_scheduler.LambdaLR(
            optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9))

    if optimize_geometry:
        optimizer_mesh = geometry.getOptimizer(learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(
            optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9))

    optimizer = torch.optim.Adam(trainable_list, lr=learning_rate_mat)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=FLAGS.batch,
        collate_fn=dataset_train.collate,
        shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(
        dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)

    # Creates a GradScaler once at the beginning of training
    for it, target in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        target = prepare_batch(target, FLAGS.train_res, 'random')

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        display_image = FLAGS.display_interval and (it % FLAGS.display_interval
                                                    == 0)
        save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
        if display_image or save_image:
            result_image, result_dict = validate_itr(
                glctx_display,
                prepare_batch(next(v_it), FLAGS.train_res, FLAGS.background),
                dataset_validate.getMesh(), geometry, opt_material, lgt, FLAGS,
                denoiser, it)
            np_result_image = result_image.detach().cpu().numpy()
            if display_image:
                util.display_image(np_result_image,
                                   title='%d / %d' % (it, FLAGS.iter))
            if save_image:
                util.save_image(
                    FLAGS.out_dir + '/' + ('img_%s_%06d.png' %
                                           (pass_name, img_cnt)),
                    np_result_image)
                img_cnt = img_cnt + 1

        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()
        if optimize_light:
            optimizer_light.zero_grad()

        # ==============================================================================================
        #  Initialize training
        # ==============================================================================================
        iter_start_time = time.time()

        # ==============================================================================================
        #  Geometry-specific training
        # ==============================================================================================
        if optimize_light:
            lgt.update_pdf()

        img_loss, reg_loss = geometry.tick(glctx, target, lgt, opt_material,
                                           image_loss_fn, it, FLAGS, denoiser)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================

        total_loss.backward()
        if FLAGS.learn_lighting and hasattr(
                lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64

        if 'kd_ks' in opt_material:
            opt_material['kd_ks'].encoder.params.grad /= 8.0

        # Optionally clip gradients
        if FLAGS.clip_max_norm > 0.0:
            if optimize_geometry:
                torch.nn.utils.clip_grad_norm_(
                    geometry.parameters() + trainable_list,
                    FLAGS.clip_max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(trainable_list,
                                               FLAGS.clip_max_norm)

        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        if optimize_light:
            optimizer_light.step()
            scheduler_light.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================

        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(
                    min=0.01
                )  # For some reason gradient dissapears if light becomes 0

        # ==============================================================================================
        #  Log & save outputs
        # ==============================================================================================
        torch.cuda.synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # Print/save log.
        if log_interval and (it % log_interval == 0):
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))

            remaining_time = (FLAGS.iter - it) * iter_dur_avg
            print(
                "iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s"
                % (it, img_loss_avg, reg_loss_avg, 0.0, iter_dur_avg * 1000,
                   util.time_to_text(remaining_time)))

    return geometry, opt_material


def demo():
    FLAGS = getFLAGS()

    FLAGS.ref_mesh = "/home/chli/chLi/NeRF/ustc_niu_merge_10"
    FLAGS.random_textures = True
    FLAGS.iter = 5000
    FLAGS.save_interval = 100
    FLAGS.texture_res = [1024, 1024]
    FLAGS.train_res = [1280, 720]
    FLAGS.batch = 1
    FLAGS.learning_rate = [0.03, 0.005]
    FLAGS.dmtet_grid = 128
    FLAGS.mesh_scale = 3.0
    FLAGS.validate = True
    FLAGS.n_samples = 8
    FLAGS.laplace_scale = 6000
    FLAGS.denoiser = "bilateral"
    FLAGS.display = [{
        "latlong": True
    }, {
        "bsdf": "kd"
    }, {
        "bsdf": "ks"
    }, {
        "bsdf": "normal"
    }]
    FLAGS.background = "white"
    FLAGS.transparency = True
    FLAGS.out_dir = "./out/ustc_niu_merge_10/"

    FLAGS.display_res = FLAGS.train_res
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()

    glctx_display = glctx if FLAGS.batch < 16 else dr.RasterizeGLContext(
    )  # Context for display

    assert os.path.isdir(FLAGS.ref_mesh)
    assert os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transform.json'))
    dataset_train = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transform.json'),
                                FLAGS,
                                examples=(FLAGS.iter + 1) * FLAGS.batch)
    dataset_validate = DatasetNERF(
        os.path.join(FLAGS.ref_mesh, 'transform.json'), FLAGS)

    lgt = light.create_trainable_env_rnd(256, scale=0.0, bias=0.5)

    denoiser = BilateralDenoiser().cuda()

    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)

    mat = initial_guess_material(geometry, True, FLAGS)

    mat['no_perturbed_nrm'] = True

    geometry, mat = optimize_mesh(denoiser,
                                  glctx,
                                  glctx_display,
                                  geometry,
                                  mat,
                                  lgt,
                                  dataset_train,
                                  dataset_validate,
                                  FLAGS,
                                  pass_idx=0,
                                  pass_name="dmtet_pass1",
                                  optimize_light=True)

    #  validate(glctx_display, geometry, mat, lgt, dataset_validate,
    #  os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS,
    #  denoiser)

    base_mesh = xatlas_uvmap(glctx_display, geometry, mat, FLAGS).clone()
    base_mesh.v_pos = base_mesh.v_pos.clone().detach().requires_grad_(True)
    mat = material.create_trainable(base_mesh.material.copy())
    geometry = DLMesh(base_mesh, FLAGS)

    os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
    light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"),
                       lgt)

    if FLAGS.transparency:
        FLAGS.layers = 8

    mat['no_perturbed_nrm'] = False

    geometry, mat = optimize_mesh(denoiser,
                                  glctx,
                                  glctx_display,
                                  geometry,
                                  mat,
                                  lgt,
                                  dataset_train,
                                  dataset_validate,
                                  FLAGS,
                                  pass_idx=1,
                                  pass_name="mesh_pass",
                                  warmup_iter=100,
                                  optimize_light=True,
                                  optimize_geometry=True)

    final_mesh = geometry.getMesh(mat)
    os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
    light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

    #  validate(glctx_display, geometry, mat, lgt, dataset_validate,
    #  os.path.join(FLAGS.out_dir, "validate"), FLAGS, denoiser)

    final_mesh = geometry.getMesh(mat)
    os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
    light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)
    return True
