import math

from keras.models import Model
import tensorflow as tf


class Cowclip(Model):
    def __init__(self, sparse_embed_dim: int, clip: float = 0, bound: float = 0, log: bool = False, log_freq: int = 100,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = clip
        self.log = log
        self.log_freq = log_freq
        self.bound = bound
        self.sparse_embed_dim = sparse_embed_dim
        self.cur_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def train_step(self, data):
        ret = dict()

        # log setting
        self.cur_step.assign_add(1)

        def should_record():
            return tf.equal(tf.math.floormod(self.cur_step, self.log_freq), 0)

        record_env = tf.summary.record_if(should_record)
        tf.summary.experimental.set_step(self.cur_step)

        # assist vars
        name_to_layer = {x.name: x for x in self.trainable_variables}
        uniq_ids, uniq_cnt = dict(), dict()
        for k, v in data[0].items():
            if k[0] != "I":
                y, _, count = tf.unique_with_counts(v)
                uniq_ids[k] = y
                uniq_cnt[k] = count

        # main
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # loss = self.compiled_loss(y, y_pred)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # clip
        name_to_gradient = {
            x.name: g for x, g in zip(self.trainable_variables, gradients)
        }
        embed_index = [
            i for i, x in enumerate(trainable_vars) if "embeddings" in x.name
        ]
        dense_index = [i for i in range(
            len(trainable_vars)) if i not in embed_index]
        embed_vars = [trainable_vars[i] for i in embed_index]
        dense_vars = [trainable_vars[i] for i in dense_index]
        embed_gradients = [gradients[i] for i in embed_index]
        dense_gradients = [gradients[i] for i in dense_index]

        # CowClip
        if self.clip > 0:
            lower_bound = self.clip * math.sqrt(self.sparse_embed_dim) * self.bound
            embed_gradients_clipped = []
            for w, g in zip(embed_vars, embed_gradients):
                if 'linear' in w.name:
                    embed_gradients_clipped.append(g)
                    continue
                prefix = "sparse_emb_"
                col_name = w.name[w.name.find(prefix) + len(prefix): w.name.find("/")]

                g_clipped = self.cow_clip(w, g, ratio=self.clip, ids=uniq_ids[col_name], cnts=uniq_cnt[col_name],
                                          min_w=lower_bound)
                embed_gradients_clipped.append(g_clipped)

            embed_gradients = embed_gradients_clipped

        gradients = embed_gradients + dense_gradients

        # update
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # =====
        # Logging
        # =====
        embed_gradients = [gradients[i] for i in embed_index]
        dense_gradients = [gradients[i] for i in dense_index]
        with record_env:
            if self.log:
                specs = None
                if hasattr(self.optimizer, 'optimizer_specs'):
                    specs = self.optimizer.optimizer_specs
                else:
                    specs = self.optimizer.param_groups
                tf.summary.scalar("lr/dense", specs[1]['optimizer']._decayed_lr('float32'))
                tf.summary.scalar("loss/loss", loss)
                tf.summary.scalar("global_norm/global", tf.linalg.global_norm(gradients))
                tf.summary.scalar("global_norm/dense", tf.linalg.global_norm(dense_gradients))
                tf.summary.scalar("global_norm/embed", tf.linalg.global_norm(embed_vars))
                tf.summary.scalar("global_norm/var_dense", tf.linalg.global_norm(dense_vars))
                tf.summary.scalar("global_norm/var_embed", tf.linalg.global_norm(embed_gradients))


                for i, (variable, gradient) in enumerate(zip(trainable_vars, gradients)):
                    name = variable.name
                    opt_index = 0 if i in embed_index else 1
                    m = specs[opt_index]["optimizer"].get_slot(variable, "m")
                    v = specs[opt_index]["optimizer"].get_slot(variable, "v")

                    layer_norm = tf.norm(variable)
                    grad_norm = tf.norm(gradient)
                    m_norm = tf.norm(m)
                    v_norm = tf.norm(v)

                    tf.summary.scalar("layer_norm/" + name, layer_norm)
                    tf.summary.scalar("grad_norm/" + name, grad_norm)
                    tf.summary.scalar("m_norm/" + name, m_norm)
                    tf.summary.scalar("v_norm/" + name, v_norm)

        self.compiled_metrics.update_state(y, y_pred)
        for m in self.metrics:
            ret[m.name] = m.result()
        return ret

    def cow_clip(self, w, g, ratio=1., ids=None, cnts=None, min_w=0.03, const=False):
        if isinstance(g, tf.IndexedSlices):
            # FIXME: This part is not tested
            values = tf.convert_to_tensor(g.values)
            clipnorm = tf.norm(tf.gather(w, g.indices), axis=-1)
        else:
            values = g
            if const:
                clipnorm = tf.constant([min_w] * g.shape[0])
            else:
                clipnorm = tf.norm(w, axis=-1)
                # bound weight norm by min_w
                clipnorm = tf.maximum(clipnorm, min_w)
            # scale by cnting
            cnts = tf.tensor_scatter_nd_update(
                tf.ones([clipnorm.shape[0]], dtype=tf.int32),
                tf.expand_dims(ids, -1),
                cnts,
            )
            clipnorm = clipnorm * tf.cast(cnts, tf.float32)

        clip_t = ratio * clipnorm
        l2sum_row = tf.reduce_sum(values * values, axis=-1)
        pred = l2sum_row > 0
        l2sum_row_safe = tf.where(pred, l2sum_row, tf.ones_like(l2sum_row))
        l2norm_row = tf.sqrt(l2sum_row_safe)
        intermediate = values * tf.expand_dims(clip_t, -1)
        g_clip = intermediate / tf.expand_dims(tf.maximum(l2norm_row, clip_t), -1)

        if isinstance(g, tf.IndexedSlices):
            return tf.IndexedSlices(g_clip, g.indices, g.dense_shape)
        else:
            return g_clip
