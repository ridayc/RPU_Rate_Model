from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import numpy as np

"""
modular_probe.py
================
Modular MNIST representation probe for E/I balanced networks.
Supports global or per-stack readouts (centroid hard-vote or softmax)
and sklearn‑based classifiers (logistic, SVM, MLP).
"""
import argparse
import time
import torch
import torchvision as tv
import torchvision.transforms as T
from network_session import build_session, session_arg_parser


# ======================================================================
# 0. STATE SNAPSHOT / RESET HELPERS
# ======================================================================
# Mirrors the fix applied to phase_trajectory_plot.py: a population's
# .rates is NOT the only stateful variable that can carry information
# from one trial into the next. Each compartment also keeps an `lrates`
# buffer and (where present) an `SST.gavg` running average. If only
# `.rates` is reset between images, those other variables keep
# accumulating image-to-image, which silently leaks information about
# every prior image into what is supposed to be an independent trial.

def _clone_state(net):
    """Snapshot every population's rates plus every compartment's
    lrates and SST.gavg (where present), across the whole network."""
    state = {}
    for p in net.populations.values():
        state[p.id] = {}
        r = p.rates
        state[p.id]["rates"] = r.clone() if isinstance(r, torch.Tensor) else r.copy()
        for c in p.compartments.values():
            state[p.id][c.id] = {}
            lr = c.lrates
            state[p.id][c.id]["lrates"] = (lr.clone() if isinstance(lr, torch.Tensor)
                                            else lr.copy())
            if c.SST is not None and c.SST.type != "pre":
                g = c.SST.gs
                state[p.id][c.id]["gs"] = (g.clone() if isinstance(g, torch.Tensor)
                                              else g.copy())
                g = c.SST.gf
                state[p.id][c.id]["gf"] = (g.clone() if isinstance(g, torch.Tensor)
                                              else g.copy())
    return state


def _restore_state(net, state):
    """Write a snapshot produced by _clone_state / _zero_state back into
    the network."""
    for p in net.populations.values():
        p.rates[:] = state[p.id]["rates"]
        for c in p.compartments.values():
            c.lrates[:] = state[p.id][c.id]["lrates"]
            if c.SST is not None and c.SST.type != "pre":
                c.SST.gs[:] = state[p.id][c.id]["gs"]
                c.SST.gf[:] = state[p.id][c.id]["gf"]


def _zero_state(net):
    """Build an all-zero snapshot with the same structure as
    _clone_state, so a 'fresh' reset zeroes rates, lrates, AND gavg --
    not just rates."""
    state = {}
    for p in net.populations.values():
        state[p.id] = {}
        r = p.rates
        state[p.id]["rates"] = (torch.zeros_like(r) if isinstance(r, torch.Tensor)
                                 else np.zeros_like(r))
        for c in p.compartments.values():
            state[p.id][c.id] = {}
            lr = c.lrates
            state[p.id][c.id]["lrates"] = (torch.zeros_like(lr) if isinstance(lr, torch.Tensor)
                                            else np.zeros_like(lr))
            if c.SST is not None and c.SST.type != "pre":
                g = c.SST.gs
                state[p.id][c.id]["gs"] = (torch.zeros_like(g) if isinstance(g, torch.Tensor)
                                              else np.zeros_like(g))
                g = c.SST.gf
                state[p.id][c.id]["gf"] = (torch.zeros_like(g) if isinstance(g, torch.Tensor)
                                              else np.zeros_like(g))
    return state


# ======================================================================
# 1. STEPPER: Raw network runner
# ======================================================================
class NetworkStepper:
    def __init__(self, session, warm_up_steps, steps_per_img, cool_down_steps, n_cycles, noise,gain,
                 reset_between=False, reset_mode='loaded', preroll_images=None, device='cpu'):
        """
        reset_between : if True, the ENTIRE network's state (every
            population's rates, plus every compartment's lrates and
            SST.gavg where present) is reset before every call to run()
            -- i.e. before every image. If False, state carries over from
            one image to the next (whatever the network is left holding
            after the previous image's full warm-up/steps/cool-down
            cycle) -- e.g. to study learning/adaptation across an ongoing
            sequence of images.

        reset_mode : only relevant when reset_between is True. Selects
            what fixed baseline every image gets reset back to:
            'loaded'  (default): the full network state captured at
                NetworkStepper construction time (i.e. however the
                network came out of session/build_session loading),
                reused for every single image.
            'fresh': an all-zero state (rates, lrates, and gavg all
                zero), reused for every image.
            'preroll': run the network through `preroll_images` image
                presentations first, with NO reset between them (same
                dynamics as reset_between=False would produce), then
                snapshot whatever state that leaves the network in and
                use THAT as the fixed reset baseline for every later
                image. This tests whether the network can still tell
                inputs apart from some generic, already-evolved starting
                configuration -- without the unbounded, ever-drifting
                state of reset_between=False, since every trial after
                the preroll starts from the exact same point.

        preroll_images : required iterable of images when
            reset_mode == 'preroll'; ignored otherwise. These images are
            only used to evolve the network into a starting state -- the
            resulting rates are discarded, not added to calibration or
            test data.
        """
        self.session = session
        self.warm_up = warm_up_steps
        self.steps = steps_per_img
        self.cool_down = cool_down_steps
        self.n_cycles = n_cycles
        self.noise = noise
        self.gain = gain
        self.reset = reset_between
        self.reset_mode = reset_mode
        self.device = device
        self.net = session.net

        self._baseline_state = None
        if self.reset:
            if reset_mode not in ("loaded", "fresh", "preroll"):
                raise ValueError(f"Unknown reset_mode '{reset_mode}', "
                                  f"expected 'loaded', 'fresh', or 'preroll'.")
            if reset_mode == "loaded":
                self._baseline_state = _clone_state(self.net)
            elif reset_mode == "fresh":
                self._baseline_state = _zero_state(self.net)
            else:  # "preroll"
                if not preroll_images:
                    raise ValueError("reset_mode='preroll' requires preroll_images "
                                      "(a non-empty iterable of images).")
                self._run_preroll(preroll_images)
                self._baseline_state = _clone_state(self.net)

    def _run_preroll(self, images):
        """
        Present `images` back-to-back with no reset between them -- same
        spirit as a reset_between=False run -- purely to evolve the
        network into a generic, already-active starting state. The rates
        produced are discarded; only the final network state matters,
        and it's captured by the caller right after this returns.
        """
        P = self.net.populations["P"]
        n_cycles = abs(self.n_cycles)
        for img in images:
            for cycle in range(n_cycles):
                P.rates.zero_()
                for _ in range(self.warm_up):
                    self.session.step()
                temp_img = (img.flatten()*self.gain).to(self.device)
                for t in range(self.steps):
                    P.rates.copy_(temp_img)
                    self.session.step()
                P.rates.zero_()
                for t in range(self.cool_down):
                    self.session.step()

    def run(self, img, pop_ids, collect_rates=True):
        P = self.net.populations["P"]
        n_cycles = abs(self.n_cycles)
        if collect_rates:
            buffers = {pid: [] for pid in pop_ids}
        else:
            buffers = None
        r0 = {}
        static = torch.zeros_like(img.flatten().to(self.device)).to(self.device)
        if (self.noise>0):
            static[:] = (self.noise) * torch.randn_like(P.rates)*self.gain
        for cycle in range(n_cycles):
            P.rates.zero_()
            for _ in range(self.warm_up):
                self.session.step()
            temp_img = (img.flatten()*self.gain).to(self.device)
            for t in range(self.steps):
                P.rates.copy_(temp_img)
                if (self.noise<0):
                    static.copy_(-(self.noise) * torch.randn_like(P.rates)*self.gain)
                if(self.noise!=0):
                    P.rates+=static
                    P.rates.clamp_(min=0)
                if collect_rates:
                    for pid in pop_ids:
                        r0[pid] = self.net.populations[pid].rates.detach().cpu().clone()
                self.session.step()
                if collect_rates:
                    for pid in pop_ids:
                        r = self.net.populations[pid].rates.detach().cpu().clone()
                        ravg = self.net.populations[pid].compartments["E_"+pid].rate_out.detach().cpu().clone()
                        tau = self.net.populations[pid].tau
                        r = (r-(1-tau)*r0[pid])#/tau
                        #r = torch.pow(torch.abs(r),0.25)*torch.sign(r)
                        r = torch.sign(r)*(torch.abs(r)>1e-2)
                        buffers[pid].append(r)
            for t in range(self.cool_down):
                P.rates.zero_()
                if collect_rates:
                    for pid in pop_ids:
                        r0[pid] = self.net.populations[pid].rates.detach().cpu().clone()
                self.session.step()
                if collect_rates:
                    for pid in pop_ids:
                        r = self.net.populations[pid].rates.detach().cpu().clone()
                        ravg = self.net.populations[pid].compartments["E_"+pid].rate_out.detach().cpu().clone()
                        tau = self.net.populations[pid].tau
                        r = (r-(1-tau)*r0[pid])#/tau
                        #r = torch.pow(torch.abs(r),0.25)*torch.sign(r)
                        r = torch.sign(r)*(torch.abs(r)>1e-2)
                        buffers[pid].append(r)

        if not collect_rates:
            return None
        return {pid: torch.stack(buffers[pid]) for pid in pop_ids}

    def run_sequence(self, img_seq, pop_ids, collect_rates=True):
        # restore to baseline if the reset option is selected
        if self.reset:
            _restore_state(self.net, self._baseline_state)
        # currently history is the only selectable option
        for i in range(len(img_seq)):
            if(i!=len(img_seq)-1):
                self.run(img_seq[i], pop_ids, collect_rates=False)
            else:
                rates = self.run(img_seq[i], pop_ids, collect_rates=collect_rates)
            # if we processed the first image we store the network state for later retrieval
            if(i==0 and not self.reset):
                self._baseline_state = _clone_state(self.net)
        # no need to reset here for the reset case as a reset call happens at the beginning of the loop
        # if not, we need to reset to the point after the first image was presented to the network
        # the idea here is to be able to use mnist images sequentially and go through img_seq as a ring buffer for the upcoming images.
        # Essentially a: what if I were to continue x images from the current and try to make a statement about this image in the past
        if not self.reset:
            _restore_state(self.net, self._baseline_state)
        return rates
                



# ======================================================================
# 2. EXTRACTORS: Raw rates -> flat 1D vector
# ======================================================================
class FeatureExtractor:
    def extract(self, raw, session):
        raise NotImplementedError

class MeanFeature(FeatureExtractor):
    def extract(self, raw, session):
        return torch.cat([raw[pid].mean(dim=0) for pid in raw])

class FinalFeature(FeatureExtractor):
    def extract(self, raw, session):
        return torch.cat([raw[pid][-1, :] for pid in raw])

class TrajectoryFeature(FeatureExtractor):
    def extract(self, raw, session):
        return torch.cat([raw[pid].flatten() for pid in raw])

class CovarianceFeature(FeatureExtractor):
    def extract(self, raw, session):
        combined = None
        for pid in raw:
            pop = session.net.populations[pid]
            if not hasattr(pop, 'size'):
                raise ValueError(f"CovarianceFeature requires spatial .size for {pid}")

            W, H, Z = pop.size
            rates = raw[pid].view(-1, W * H, Z)
            mu = rates.mean(dim=0, keepdim=True)
            centered = rates - mu
            cov = torch.einsum('tli,tlj->lij', centered, centered) / (rates.shape[0] - 1)
            idx_i, idx_j = torch.triu_indices(Z, Z, offset=1)
            upper = cov[:, idx_i, idx_j]  # (WH, D)

            if combined is None:
                combined = upper
            else:
                combined = torch.cat([combined, upper], dim=1)

        if combined is None:
            raise ValueError("No spatial populations found.")
        return combined.flatten()


# ======================================================================
# 3. READOUTS
# ======================================================================
class Readout:
    def fit(self, calib_features, calib_labels):
        raise NotImplementedError
    def predict(self, test_feature):
        raise NotImplementedError

# ----- centroid‑based readouts (unchanged) -----
class GlobalCentroidReadout(Readout):
    def __init__(self, metric='euclidean',nclass=10):
        self.metric = metric
        self.nclass = nclass
        self.centroids = None

    def fit(self, calib_features, calib_labels):
        D = calib_features[0].shape[0]
        sums = torch.zeros(self.nclass, D, dtype=torch.float64)
        counts = torch.zeros(self.nclass, dtype=torch.float64)
        for feat, lbl in zip(calib_features, calib_labels):
            sums[lbl] += feat.double()
            counts[lbl] += 1
        self.centroids = (sums / counts.unsqueeze(1).clamp(min=1)).float()

    def predict(self, test_feature):
        x = test_feature.float().unsqueeze(0)
        if self.metric == 'euclidean':
            dists = torch.cdist(x, self.centroids, p=2).squeeze(0)
            return int(torch.argmin(dists).item())
        else:
            x_n = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
            c_n = self.centroids / self.centroids.norm(dim=1, keepdim=True).clamp(min=1e-12)
            sims = (x_n @ c_n.T).squeeze(0)
            return int(torch.argmax(sims).item())

class StackCentroidReadout(Readout):
    def __init__(self, spatial_shape, metric='euclidean', nclass=10):
        self.W, self.H, self.Z = spatial_shape
        self.L = self.W * self.H
        self.metric = metric
        self.nclass = nclass
        self.centroids = None

    def fit(self, calib_features, calib_labels):
        sums = torch.zeros(self.L, self.nclass, self.Z, dtype=torch.float64)
        counts = torch.zeros(self.L, self.nclass, dtype=torch.float64)
        for feat, lbl in zip(calib_features, calib_labels):
            stack_feat = feat.view(self.L, self.Z)
            sums[:, lbl, :] += stack_feat.double()
            counts[:, lbl] += 1
        self.centroids = (sums / counts.unsqueeze(-1).clamp(min=1)).float()

    def predict(self, test_feature):
        x = test_feature.float().view(self.L, self.Z)
        c = self.centroids
        if self.metric == 'euclidean':
            dists = torch.cdist(x.unsqueeze(1), c, p=2).squeeze(1)
            votes = torch.argmin(dists, dim=1)
        else:
            x_n = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
            c_n = c / c.norm(dim=2, keepdim=True).clamp(min=1e-12)
            sims = torch.einsum('ld,lcd->lc', x_n, c_n)
            votes = torch.argmax(sims, dim=1)
        counts = torch.bincount(votes, minlength=10)
        return int(torch.argmax(counts).item())

class StackSoftmaxReadout(Readout):
    def __init__(self, spatial_shape, temperature=1.0, metric='euclidean', nclass=10):
        self.W, self.H, self.Z = spatial_shape
        self.L = self.W * self.H
        self.temp = temperature
        self.metric = metric
        self.nclass = nclass
        self.centroids = None

    def fit(self, calib_features, calib_labels):
        sums = torch.zeros(self.L, self.nclass, self.Z, dtype=torch.float64)
        counts = torch.zeros(self.L, self.nclass, dtype=torch.float64)
        for feat, lbl in zip(calib_features, calib_labels):
            stack_feat = feat.view(self.L, self.Z)
            sums[:, lbl, :] += stack_feat.double()
            counts[:, lbl] += 1
        self.centroids = (sums / counts.unsqueeze(-1).clamp(min=1)).float()

    def predict(self, test_feature):
        x = test_feature.float().view(self.L, self.Z)
        c = self.centroids
        if self.metric == 'euclidean':
            dists = torch.cdist(x.unsqueeze(1), c, p=2).squeeze(1)
            scores = -dists / self.temp
        else:
            x_n = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
            c_n = c / c.norm(dim=2, keepdim=True).clamp(min=1e-12)
            scores = torch.einsum('ld,lcd->lc', x_n, c_n) / self.temp
        probs = torch.softmax(scores, dim=1)
        total_vote = probs.sum(dim=0)
        return int(torch.argmax(total_vote).item())

# ----- sklearn‑based readouts (new) -----
class SklearnLogisticReadout(Readout):
    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.clf = None

    def fit(self, calib_features, calib_labels):
        X = torch.stack(calib_features).numpy()
        y = np.array(calib_labels)
        self.clf = LogisticRegression(
            penalty='l2', C=self.C, solver='lbfgs',
            max_iter=self.max_iter, multi_class='multinomial'
        )
        self.clf.fit(X, y)

    def predict(self, test_feature):
        x = test_feature.numpy().reshape(1, -1)
        return int(self.clf.predict(x)[0])

class SklearnSVMReadout(Readout):
    def __init__(self, C=1.0, kernel='rbf', max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.clf = None

    def fit(self, calib_features, calib_labels):
        X = torch.stack(calib_features).numpy()
        y = np.array(calib_labels)
        self.clf = SVC(C=self.C, kernel=self.kernel, max_iter=self.max_iter, probability=False)
        self.clf.fit(X, y)

    def predict(self, test_feature):
        x = test_feature.numpy().reshape(1, -1)
        return int(self.clf.predict(x)[0])

class SklearnMLPReadout(Readout):
    def __init__(self, hidden_sizes=(100,), alpha=1e-4, max_iter=1000):
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.clf = None

    def fit(self, calib_features, calib_labels):
        X = torch.stack(calib_features).numpy()
        y = np.array(calib_labels)
        self.clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=self.hidden_sizes,
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping=False
            )
        )
        self.clf.fit(X, y)

    def predict(self, test_feature):
        x = test_feature.numpy().reshape(1, -1)
        return int(self.clf.predict(x)[0])

class MultiCentroidReadout(Readout):
    def __init__(self, n_centroids_per_class=5, metric='euclidean', nclass=10, temperature=1.0, pooling='hard'):
        """
        n_centroids_per_class : int, number of centroids to learn per class
        metric : 'euclidean' or 'cosine'
        temperature : float, used only if pooling='soft'
        pooling : 'hard' or 'soft'
        """
        self.K = n_centroids_per_class
        self.metric = metric
        self.nclass = nclass
        self.temp = temperature
        self.pooling = pooling
        self.centroids = None   # will be a list of tensors, one per class, each of shape [K, D]

    def fit(self, calib_features, calib_labels):
        # Group features by label
        class_feats = {i: [] for i in range(self.nclass)}
        for feat, lbl in zip(calib_features, calib_labels):
            class_feats[lbl].append(feat.numpy())   # convert to numpy for kmeans

        self.centroids = []
        for lbl in range(self.nclass):
            X = np.array(class_feats[lbl])
            if len(X) == 0:
                # fallback: use global mean if no samples (shouldn't happen with MNIST)
                X = np.random.randn(self.K, calib_features[0].shape[0])
            if len(X) < self.K:
                # if fewer samples than K, duplicate with noise
                k = min(self.K, len(X))
                km = KMeans(n_clusters=k, random_state=0).fit(X)
                centroids = km.cluster_centers_
                # duplicate remaining centroids with small random noise
                while len(centroids) < self.K:
                    centroids = np.vstack([centroids, centroids[-1] + 1e-6 * np.random.randn(centroids.shape[1])])
            else:
                km = KMeans(n_clusters=self.K, random_state=0).fit(X)
                centroids = km.cluster_centers_
            self.centroids.append(torch.tensor(centroids, dtype=torch.float32))

    def predict(self, test_feature):
        x = test_feature.float().unsqueeze(0)   # [1, D]
        class_scores = []
        for c_centroids in self.centroids:       # c_centroids: [K, D]
            if self.metric == 'euclidean':
                # distances from x to all K centroids of this class
                dists = torch.cdist(x, c_centroids, p=2)   # [1, K]
                scores = -dists / self.temp               # convert to similarity
            else:  # cosine
                x_n = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
                c_n = c_centroids / c_centroids.norm(dim=1, keepdim=True).clamp(min=1e-12)
                scores = x_n @ c_n.T / self.temp          # [1, K]

            if self.pooling == 'hard':
                # each centroid votes for its best class (which is this class because we are iterating)
                # For hard vote, we just count the number of centroids that assign the input to this class.
                # Here we need to know the nearest centroid's label, but since we are computing per class,
                # we can simply count how many centroids are closer than the nearest centroid of other classes.
                # Simpler: compute distances to all centroids of all classes, then assign nearest centroid,
                # then count per class. Let's do that instead.
                pass  # we'll implement a simpler global approach below

        # Better: compute all centroids at once and then pool
        # Concatenate all centroids: [10*K, D]
        all_centroids = torch.cat(self.centroids, dim=0)   # [nclass*K, D]
        if self.metric == 'euclidean':
            dists = torch.cdist(x, all_centroids, p=2)     # [1, nclass*K]
            scores = -dists / self.temp                    # [1, nclass*K]
        else:
            x_n = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
            all_n = all_centroids / all_centroids.norm(dim=1, keepdim=True).clamp(min=1e-12)
            scores = x_n @ all_n.T / self.temp             # [1, nclass*K]

        # Reshape scores to [nclass, K]
        scores_per_class = scores.view(self.nclass, self.K)         # [nclass, K]

        if self.pooling == 'hard':
            # For each centroid, pick the class it belongs to (we already have class index via row)
            # But we need to know which centroid is nearest overall.
            # Instead, we can take the argmin of distances (or argmax of scores) over all centroids,
            # which gives the centroid index. Then the class is centroid_index // K.
            best_indices = torch.argmax(scores, dim=1)      # [1]
            best_class = best_indices // self.K
            return int(best_class.item())
        elif self.pooling == 'soft':
            # Compute softmax over all centroids? Or per class?
            # Option: sum softmax over centroids per class, then take argmax.
            # We can compute softmax over the flattened scores (across all centroids) – that gives
            # a probability distribution over all centroids. Then sum probabilities per class.
            probs_all = torch.softmax(scores, dim=1)        # [1, 10*K]
            probs_class = probs_all.view(10, self.K).sum(dim=1)  # [10]
            return int(torch.argmax(probs_class).item())
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")


# ======================================================================
# 4. ORCHESTRATOR
# ======================================================================

def run_sequence_pipeline(session, calib_loader, test_loader, pop_ids,
                          stepper, extractor, readout,
                          history,x_seq, n_calib, n_test, paired, n_skip_calib=0, label="seq"):
    """
    Calibrate and test on look‑ahead sequence tasks.

    history : int, number of images to look forward.
              history=1 → standard single‑image classification.
              history=2 → present [img_t, img_{t+1}], predict label_{t+x_seq}.
              history=3 → present [img_t, img_{t+1}, img_{t+2}], predict label_{t+x_seq}.
    """
    print(f"[{label}] Building stream from calibration loader (history={history})...")

    # ------------------------------------------------------------
    # 1. Materialise the calibration stream as a list of (img, lbl)
    # ------------------------------------------------------------
    calib_stream = []
    for imgs, lbls in calib_loader:
        calib_stream.append((imgs[0, 0] * 255., int(lbls[0])))

    # Skip the first n_skip_calib images (used for preroll, if any)
    start_idx = n_skip_calib
    total_calib = len(calib_stream)

    # We need at least `history` images to form a complete window
    max_start_idx = total_calib -1#- history
    #available_windows = max(0, max_start_idx - start_idx + 1)
    available_windows = max(0, max_start_idx+1)

    # Cap the number of calibration samples
    n_calib = min(n_calib, available_windows)
    print(f"[{label}] Calibrating on {n_calib} windows...")

    # ------------------------------------------------------------
    # 2. Calibration loop
    # ------------------------------------------------------------
    calib_features, calib_labels = [], []
    t0 = time.time()
    for i in range(0, n_calib):
        idx = (start_idx + i) % total_calib
        # Collect the window of images
        img_seq = [calib_stream[(idx + j)% total_calib][0] for j in range(history)]

        # Run the sequence – this uses YOUR run_sequence with clone/restore
        raw = stepper.run_sequence(img_seq, pop_ids, collect_rates=True)

        # Extract the feature from the LAST image (which now contains the whole context)
        feat = extractor.extract(raw, session)
        calib_features.append(feat)

        # The label is the x_seq'th image in the window
        if(not paired):
            calib_labels.append(calib_stream[(idx+x_seq)% total_calib][1])
        else:
            joint_label = 0
            for step in range(x_seq+1):
                # Grab the digit at the current step in the sequence
                digit = calib_stream[(idx + step) % total_calib][1]
                # Shift the existing joint_label left by one decimal place and add the new digit
                joint_label = (joint_label * 10) + digit

            calib_labels.append(joint_label)

        if (idx - start_idx + 1) % 500 == 0:
            speed = (idx - start_idx + 1) / (time.time() - t0)
            print(f"  Calib {idx - start_idx + 1}/{n_calib}  {speed:.1f} img/s")

    readout.fit(calib_features, calib_labels)
    print(f"  Calibration done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------
    # 3. Materialise the test stream
    # ------------------------------------------------------------
    test_stream = []
    for imgs, lbls in test_loader:
        test_stream.append((imgs[0, 0] * 255., int(lbls[0])))

    total_test = len(test_stream)
    max_test_idx = total_test -1#- history
    n_test = min(n_test, max_test_idx + 1)

    print(f"[{label}] Evaluating on {n_test} windows...")
    correct = 0
    t0 = time.time()
    for i in range(n_test):
        idx = i % total_test
        img_seq = [test_stream[(idx + j)% total_test][0] for j in range(history)]
        raw = stepper.run_sequence(img_seq, pop_ids, collect_rates=True)
        feat = extractor.extract(raw, session)
        pred = readout.predict(feat)
        if(not paired):
            true_label = test_stream[(idx + x_seq)% total_test][1]
        else:
            joint_label = 0
            for step in range(x_seq+1):
                # Grab the digit at the current step in the sequence
                digit = test_stream[(idx + step) % total_test][1]
                # Shift the existing joint_label left by one decimal place and add the new digit
                joint_label = (joint_label * 10) + digit

            true_label = joint_label
        correct += (pred == true_label)

        if (idx + 1) % 100 == 0:
            acc = correct / (idx + 1)
            print(f"  Eval {idx + 1}/{n_test}  acc={acc:.3f}  {idx/(time.time()-t0):.1f} img/s")

    return correct / n_test


# ======================================================================
# 5. MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Modular representation probe for E/I networks.",
        parents=[session_arg_parser()],
    )
    parser.add_argument("--n-calib", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--steps-per-img", type=int, default=150)
    parser.add_argument("--warm-up", type=int, default=6)
    parser.add_argument("--cool-down", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=1,
                        help="Positive: avg last step per cycle. Negative: avg first step per cycle.")
    parser.add_argument("--populations", nargs="+", default=["E"])
    parser.add_argument("--reset", action="store_true",
                        help="Reset the ENTIRE network's state (every population's rates, "
                             "plus every compartment's lrates and SST.gavg where present) "
                             "before every image, not just the populations being probed. "
                             "Without this flag, state -- including running averages like "
                             "lrates/gavg -- silently carries over from one image to the "
                             "next across the whole calibration and test loop.")
    parser.add_argument("--reset-mode", choices=["loaded", "fresh", "preroll"], default="loaded",
                        help="Only used when --reset is given. 'loaded' (default): reset "
                             "every image to the full network state as it was when the "
                             "stepper was constructed (i.e. straight out of session "
                             "loading). 'fresh': reset every image to an all-zero state "
                             "(rates, lrates, and gavg all zero). 'preroll': run the "
                             "network through --preroll-images image presentations first "
                             "with no reset between them (same dynamics as --reset NOT "
                             "being used), then fix whatever state that produces as the "
                             "reset baseline for every later image -- tests whether the "
                             "network can still discriminate digits starting from some "
                             "generic, already-evolved state, without the unbounded "
                             "drift of leaving --reset off entirely.")
    parser.add_argument("--preroll-images", type=int, default=200,
                        help="Number of images to run through the network (no reset, "
                             "discarded afterward) to produce the fixed baseline state "
                             "when --reset-mode preroll is used. Drawn from the "
                             "calibration set, disjoint from the --n-calib images "
                             "calibration actually fits on.")
    parser.add_argument("--extractor", choices=["mean", "final", "trajectory", "covariance"],
                        default="mean")
    # Extended readout choices
    parser.add_argument("--readout",
                        choices=["global-centroid", "stack-centroid", "stack-softmax",
                                 "logistic", "svm", "mlp", "multi-centroid"],
                        default="global-centroid",
                        help="Readout type. For sklearn classifiers, features are used directly.")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--temperature", type=float, default=1.0)
    # New hyperparameters for sklearn classifiers
    parser.add_argument("--C", type=float, default=1.0,
                        help="Regularization strength (inverse) for logistic and SVM.")
    parser.add_argument("--svm-kernel", choices=["linear", "poly", "rbf", "sigmoid"], default="rbf",
                        help="Kernel for SVM.")
    parser.add_argument("--mlp-hidden", type=str, default="100",
                        help="Comma‑separated hidden layer sizes, e.g. '100,50'.")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Maximum iterations for sklearn optimizers.")
    parser.add_argument("--n-centroids", type=int, default=5,
                        help="Number of centroids per class (only for multi-centroid readout)")
    parser.add_argument("--pooling", choices=["hard","soft"], default="hard",
                        help="Pooling strategy for multi-centroid readout")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Gaussin noise injection to the mnist digits. Value is interpreted as std. Interpreted as static noise per image is >=0 else interpreted as time-varying noise per image.")
    parser.add_argument("--input-gain", type=float, default=1.0,
                        help="Amplification (or reduction) of the input signal.")
    parser.add_argument("--history", type=int, default=1,
                        help="Present history images and use the last image to predict the first. History of one corresponds to standard prediction.")
    parser.add_argument("--x-seq", type=int, default=0,
                        help="Predict the x-th image in the sequence based on the network state after history image presentations.")
    parser.add_argument("--paired", action="store_true",
                        help="Predict a sequence of x-seq images base on the output for the last image.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse hidden layer sizes
    if args.mlp_hidden:
        hidden_sizes = tuple(int(x) for x in args.mlp_hidden.split(',') if x)
    else:
        hidden_sizes = (100,)

    args.freeze = True
    torch.manual_seed(args.seed)

    print("[init] Building session (freeze=True)")
    session = build_session(args)
    device = session.net.device

    # ---- PATCHED: Compute spatial shape dynamically ----
    spatial_shape = None
    if args.readout.startswith("stack"):
        total_Z = 0
        W, H = None, None
        total_steps = abs(args.cycles) * args.steps_per_img

        for pid in args.populations:
            if pid not in session.net.populations:
                continue
            pop = session.net.populations[pid]
            if hasattr(pop, 'size'):
                w, h, z = pop.size
                if W is None:
                    W, H = w, h
                elif (w, h) != (W, H):
                    raise ValueError(f"Population {pid} grid ({w},{h}) != ({W},{H})")

                if args.extractor in ["mean", "final"]:
                    dim_per_pop = z
                elif args.extractor == "trajectory":
                    dim_per_pop = total_steps * z
                elif args.extractor == "covariance":
                    dim_per_pop = z * (z - 1) // 2
                else:
                    raise ValueError(f"Unknown extractor {args.extractor}")
                total_Z += dim_per_pop

        if W is None:
            raise ValueError("No spatial population found for stack readout.")
        spatial_shape = (W, H, total_Z)
        print(f"[init] Stack readout: grid {W}x{H}, total feature depth Z={total_Z} "
              f"(extractor={args.extractor}, total_steps={total_steps})")

    transform = T.Compose([T.ToTensor()])
    calib_set = tv.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = tv.datasets.MNIST("./data", train=False, download=True, transform=transform)
    calib_loader = torch.utils.data.DataLoader(calib_set, batch_size=1, shuffle=True,
                                               generator=torch.Generator().manual_seed(args.seed))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True,
                                              generator=torch.Generator().manual_seed(args.seed+1))

    # If using preroll, draw its images from the FRONT of the (shuffled)
    # calib_loader sequence, then skip over them when the real
    # calibration loop runs below -- so preroll and calibration images
    # are guaranteed disjoint, and preroll never accidentally doubles as
    # a calibration sample.
    preroll_images = None
    n_skip_calib = 0
    if args.reset and args.reset_mode == "preroll":
        print(f"[init] Drawing {args.preroll_images} preroll images "
              f"(disjoint from calibration set)...")
        preroll_images = []
        for i, (imgs, lbls) in enumerate(calib_loader):
            if i >= args.preroll_images:
                break
            preroll_images.append(imgs[0, 0] * 255.)
        n_skip_calib = len(preroll_images)

    stepper = NetworkStepper(
        session=session,
        warm_up_steps=args.warm_up,
        steps_per_img=args.steps_per_img,
        cool_down_steps=args.cool_down,
        noise = args.noise,
        gain = args.input_gain,
        n_cycles=args.cycles,
        reset_between=args.reset,
        reset_mode=args.reset_mode,
        preroll_images=preroll_images,
        device=device,
    )

    extractor = {
        "mean": MeanFeature(),
        "final": FinalFeature(),
        "trajectory": TrajectoryFeature(),
        "covariance": CovarianceFeature(),
    }[args.extractor]

    if(args.paired):
        nclass = 10**(args.x_seq+1)
    else:
        nclass = 10
    # Instantiate readout based on type
    if args.readout == "global-centroid":
        readout = GlobalCentroidReadout(metric=args.metric,nclass=nclass)
    elif args.readout == "stack-centroid":
        readout = StackCentroidReadout(spatial_shape, metric=args.metric,nclass=nclass)
    elif args.readout == "stack-softmax":
        readout = StackSoftmaxReadout(spatial_shape, temperature=args.temperature, metric=args.metric,nclass=nclass)
    elif args.readout == "logistic":
        readout = SklearnLogisticReadout(C=args.C, max_iter=args.max_iter)
    elif args.readout == "svm":
        readout = SklearnSVMReadout(C=args.C, kernel=args.svm_kernel, max_iter=args.max_iter)
    elif args.readout == "mlp":
        readout = SklearnMLPReadout(hidden_sizes=hidden_sizes, max_iter=args.max_iter)
    elif args.readout == "multi-centroid":
        readout = MultiCentroidReadout(
            n_centroids_per_class=args.n_centroids,
            metric=args.metric,nclass=nclass,
            temperature=args.temperature,
            pooling=args.pooling
        )
    else:
        raise ValueError(f"Unknown readout: {args.readout}")

    acc = run_sequence_pipeline(
        session=session,
        calib_loader=calib_loader,
        test_loader=test_loader,
        pop_ids=args.populations,
        stepper=stepper,
        extractor=extractor,
        readout=readout,
        history=args.history,
        x_seq=args.x_seq,
        paired=args.paired,
        n_calib=args.n_calib,
        n_test=args.n_test,
        n_skip_calib=n_skip_calib,
        label=f"{args.extractor}+{args.readout}"
    )

    print("\n" + "=" * 50)
    print(f"Final Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 50)
    session.close()

if __name__ == "__main__":
    main()