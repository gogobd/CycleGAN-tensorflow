from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import sys

from module import discriminator
from module import generator_resnet
from module import mae_criterion
from module import abs_criterion
from module import generator_unet
from module import sce_criterion
from utils import load_test_data
from utils import ImagePool
from utils import save_images
from utils import ImageMemMap
import random


class cyclegan(object):

    def __init__(self, sess, args):
        self.sess = sess
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.stdscr = args.stdscr

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(
            tf.float32,
            [
                None,
                self.image_size,
                self.image_size,
                self.input_c_dim + self.output_c_dim,
            ],
            name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[
            :, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 0

        if args.continue_train and self.load(args.checkpoint_dir):
            print(" [*] Checkpoint Load SUCCESS")
        else:
            print(" [!] Checkpoint Load failed...")

        start_epoch = 0
        if args.continue_train >= 0:
            start_epoch = args.continue_train
        # elif args.continue_train < 0:
        #     start_epoch = tf.get_variable("epoch", shape=[0])

        # Prepare images for fast loading
        image_mem_map_A = ImageMemMap(
            os.path.join(self.dataset_dir, 'trainA.npy'),
            os.path.join(self.dataset_dir, 'trainA', '*'),
            load_size=args.load_size,
            fine_size=args.fine_size,
        )
        image_mem_map_B = ImageMemMap(
            os.path.join(self.dataset_dir, 'trainB.npy'),
            os.path.join(self.dataset_dir, 'trainB', '*'),
            load_size=args.load_size,
            fine_size=args.fine_size,
        )

        minsize = min(
            min(
                len(image_mem_map_A),
                len(image_mem_map_B)
            ),
            args.train_size
        )

        old_time = time.time()
        new_epoch_time = old_epoch_time = time.time()
        for epoch in range(start_epoch, args.epoch):

            batch_idxs_A = list(range(len(image_mem_map_A)))
            batch_idxs_B = list(range(len(image_mem_map_B)))

            random.shuffle(batch_idxs_A)
            random.shuffle(batch_idxs_B)

            if epoch < args.epoch_step:
                lr = args.lr
            else:
                lr = args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)

            print("Starting Epoch [%2d] lr: %1.8f" % (epoch, lr,))

            for idx in range(minsize):

                img_A = image_mem_map_A.get_image(batch_idxs_A[idx])
                img_B = image_mem_map_B.get_image(batch_idxs_B[idx])
                batch_images = [np.concatenate((img_A, img_B), axis=2)]
                batch_images = np.array(batch_images).astype(np.float32)
                # (1, 512, 512, 6)

                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                new_time = time.time()
                print(("Epoch: [%2d] [%4d/%4d] counter: %4d time: %4.4f " % (
                    epoch, idx, minsize, counter, new_time-old_time,)))

                old_time = new_time

                c = self.stdscr.getch()
                if c == ord('s') or np.mod(counter, args.save_freq) == args.save_freq - 1:
                    print("Save: save checkpoint")
                    self.save(args.checkpoint_dir, counter)
                elif c == ord('t') or np.mod(counter, args.print_freq) == args.print_freq - 2:
                    print("Test: sample_model")
                    self.sample_model(args, epoch, idx)
                elif c == ord('p'):
                    print("Pause! Press [ENTER]")
                    self.stdscr.getstr(0, 0, 1)
                    self.stdscr.clear()
                elif c == ord('q'):
                    print("QUIT!")
                    sys.exit()
                elif c != -1:
                    print("(s)ave (t)est (p)ause (q)uit")

            del batch_images
            new_epoch_time = time.time()
            print("Epoch: [%2d] DONE. lr: %1.8f time: %4.4f" % (
                epoch, lr, new_epoch_time - old_epoch_time,))
            old_epoch_time = new_epoch_time

    def save(self, checkpoint_dir, step):
        print(" [*] Writing checkpoint...")
        model_name = "cyclegan.model"
        dataset_name = os.path.split(self.dataset_dir)[-1]
        model_dir = "%s_%s_%s_%s" % (dataset_name, self.image_size, self.options.gf_dim, self.options.df_dim) # ngf ndf
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print("> dir: " + checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        dataset_name = os.path.split(self.dataset_dir)[-1]
        model_dir = "%s_%s_%s_%s" % (dataset_name, self.image_size, self.options.gf_dim, self.options.df_dim) # ngf ndf
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        print("> dir: " + checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, args, epoch, idx):
        dataA = glob(os.path.join(self.dataset_dir, 'testA', '*'))
        dataB = glob(os.path.join(self.dataset_dir, 'testB', '*'))

        np.random.shuffle(dataA)
        np.random.shuffle(dataB)

        batch_files = [dataA[0], dataB[0]]

        sample_images = []
        for batch_file in batch_files:
            sample_image = load_test_data(
                batch_file,
                args.fine_size,
            )
            sample_images.append(sample_image)

        sample_images = [np.concatenate((sample_images[0], sample_images[1]), axis=2)]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
        save_images(fake_A, [1, 1],
                    './{}/A_{:04d}_{:06d}.jpg'.format(args.sample_dir, epoch, idx))
        save_images(fake_B, [1, 1],
                    './{}/B_{:04d}_{:06d}.jpg'.format(args.sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob(os.path.join(
                '.',
                'datasets',
                self.dataset_dir,
                'testA',
                '*.*'
                )
            )
        elif args.which_direction == 'BtoA':
            sample_files = glob(os.path.join(
                '.',
                'datasets',
                self.dataset_dir,
                'testB',
                '*.*'
                )
            )
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Checkpoint Load SUCCESS")
        else:
            print(" [!] Checkpoint Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")

        index.close()
