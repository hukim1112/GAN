import tensorflow as tf
from .layers import get_upsample_blocks, get_downsample_blocks, conditioning_augmentation


class Stack_GAN(tf.keras.Model):
    def __init__(self, config):
        self.stage1 = Stage1(config['stage1'])
        self.Stage2 = Stage2(config['stage2'])

    def call(self, embeddings, real_sHxsW, real_lHxlW):
        stage1_output, fake_sHxsW = self.stage1(embeddings, real_sHxsW)
        stage2_output, fake_lHxlW = self.Stage2(
            embeddings, fake_sHxsW, real_lHxlW)
        return stage1_output, stage2_output, fake_lHxlW


class Stage1(tf.keras.Model):
    def __init__(self, config):
        self.config = config
        self.conditioning_augmentation = conditioning_augmentation()
        self.upsample_blocks = get_upsample_blocks(self.config['backbone'])
        self.downsample_blocks = get_downsample_blocks(self.config['backbone'])

        self.embedding_compressor = tf.layers.Dense(
            config['N_d'], activation='relu')
        self.image_text_feature_combinator = tf.layers.Conv2D(
            64, [1, 1], activation='relu')
        self.classifier = tf.Dense(1, activation='sigmoid')

    def call(embeddings, real_sHxsW):
        # embedding \include N x (padded)tokens x dimensions
        c_0 = self.conditioning_augmentation(embeddings)  # N x N_g
        noise = tf.random.normal(
            shape=[c_0.shape[0], self.config['N_z']])  # N x N_z
        input_noise = tf.concat([c_0, noise], 1)  # N x (N_g + N_z)
        fake_sHxsW = self.upsample_blocks(input_noise)  # N x H x W x 3
        fake_spatial_feature = self.downsample_blocks(
            fake_sHxsW)  # N x M_d x M_d x D
        real_spatial_feature = self.downsample_blocks(
            real_sHxsW)  # N x M_d x M_d x D

        # compress and spatially replicate
        compressed_embeddings = self.embedding_compressor(embeddings)
        replicated = tf.tile(compressed_embeddings, [
                             1, self.config['M_d'] * self.config['M_d']])
        spatially_replicated = tf.reshape(
            replicated, [compressed_embeddings.shape[0], self.config['M_d'], self.config['M_d'], -1])  # N x M_d x M_d x N_d

        real = self.image_text_feature_combinator(
            tf.concat([fake_spatial_feature, spatially_replicated], axis=3))
        fake = self.image_text_feature_combinator(
            tf.concat([real_spatial_feature, spatially_replicated], axis=3))

        # (N+N') x M_d x M_d x 64
        output_tesnor = tf.concat([real, fake], axis=0)
        return self.classifier(output_tesnor), fake_sHxsW


class Stage2(tf.keras.Model):
    def __init__(self, config):
        self.config = config
        self.conditioning_augmentation = conditioning_augmentation()
        self.upsample_blocks = get_upsample_blocks(self.config['backbone'])
        self.downsample_blocks = get_downsample_blocks(self.config['backbone'])

        self.embedding_compressor = tf.layers.Dense(
            self.config['N_d'], activation='relu')
        self.image_text_feature_combinator = tf.layers.Conv2D(
            64, [1, 1], activation='relu')
        self.classifier = tf.Dense(1, activation='sigmoid')

    def call(embeddings, fake_sHxsW, real_lHxlW):
        fake_spatial_feature = self.downsample_blocks(
            fake_sHxsW)  # N x M_d x M_d x D
        compressed_embeddings = self.embedding_compressor(embeddings)
        c = self.conditioning_augmentation(compressed_embeddings)  # N x N_g
        replicated = tf.tile(c, [1, self.config['M_d'] * self.config['M_d']])
        spatially_replicated = tf.reshape(
            replicated, [-1, self.config['M_d'], self.config['M_d']])  # N x M_d x M_d x N_d
        fake_lHxlW = self.upsample_blocks(
            tf.concat([fake_spatial_feature, spatially_replicated], axis=3))

        fake_spatial_feature = self.downsample_blocks(
            fake_lHxlW)  # N x M_d x M_d x D
        real_spatial_feature = self.downsample_blocks(
            real_lHxlW)  # N x M_d x M_d x D

        real = self.image_text_feature_combinator(
            tf.concat([fake_spatial_feature, spatially_replicated], axis=3))
        fake = self.image_text_feature_combinator(
            tf.concat([real_spatial_feature, spatially_replicated], axis=3))

        # (N+N') x M_d x M_d x 64
        output_tesnor = tf.concat([real, fake], axis=0)
        return self.classifier(output_tesnor), fake_lHxlW
