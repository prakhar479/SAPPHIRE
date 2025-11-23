from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import tempfile
import shutil
import sys
import logging
import numpy as np
from annoy import AnnoyIndex

# logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('demo')

# optional heavy deps will be imported lazily; set placeholders
_TORCH_AVAILABLE = False
_LIBROSA_AVAILABLE = False
torch = None
librosa = None
sf = None
try:
    import sklearn  # noqa: F401
except Exception:
    pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEMO_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(DEMO_DIR, 'index.ann')
META_PATH = os.path.join(DEMO_DIR, 'metadata.json')
EMB_PATH = os.path.join(DEMO_DIR, 'embeddings.npy')
INDEX_META_PATH = os.path.join(DEMO_DIR, 'index_meta.json')

app = Flask(__name__, template_folder='templates', static_folder='static')


def load_metadata():
    # Prefer explicit metadata file in demo folder
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            return json.load(f)
    # If there's a CSV/parquet with song metadata in the embedder folder, build metadata mapping
    # look in multiple embedder folders (mirex then top40)
    candidates = [
        os.path.join(BASE_DIR, 'models', 'embedder_mirex', 'embeddings'),
        os.path.join(BASE_DIR, 'models', 'embedder_top40', 'embeddings'),
    ]
    for embed_folder in candidates:
        csv_path = os.path.join(embed_folder, 'song_embeddings.csv')
        pqt_path = os.path.join(embed_folder, 'song_embeddings.parquet')
        if os.path.exists(csv_path) or os.path.exists(pqt_path):
            try:
                import pandas as pd
                if os.path.exists(pqt_path):
                    df = pd.read_parquet(pqt_path)
                else:
                    df = pd.read_csv(csv_path)
                meta = {}
                # create index-based metadata mapping (index -> dict)
                for i, row in enumerate(df.itertuples(index=False)):
                    # try to access common fields
                    try:
                        info = {
                            'track_id': getattr(row, 'track_id', None),
                            'mood_cluster': getattr(row, 'mood_cluster', None),
                            'mood_category': getattr(row, 'mood_category', None),
                            'dataset': getattr(row, 'dataset', None),
                        }
                    except Exception:
                        info = {}
                    meta[str(i)] = info
                return meta
            except Exception:
                return {}
    return {}


def load_embeddings():
    # If a prepared numpy exists in demo, use it
    if os.path.exists(EMB_PATH):
        return np.load(EMB_PATH)
    # Look for standard locations (numpy, parquet or csv) used in this repo
    candidates = [
        os.path.join(BASE_DIR, 'models', 'embedder_mirex', 'embeddings'),
        os.path.join(BASE_DIR, 'models', 'embedder_top40', 'embeddings'),
    ]
    for embed_folder in candidates:
        np_path = os.path.join(embed_folder, 'embeddings.npy')
        pqt_path = os.path.join(embed_folder, 'song_embeddings.parquet')
        csv_path = os.path.join(embed_folder, 'song_embeddings.csv')
        if os.path.exists(np_path):
            return np.load(np_path)
    # parquet or csv — read with pandas and extract emb_* columns
    if os.path.exists(pqt_path) or os.path.exists(csv_path):
        try:
            import pandas as pd
            if os.path.exists(pqt_path):
                df = pd.read_parquet(pqt_path)
            else:
                df = pd.read_csv(csv_path)
            # select columns starting with 'emb_'
            emb_cols = [c for c in df.columns if str(c).startswith('emb_')]
            if not emb_cols:
                return None
            arr = df[emb_cols].to_numpy(dtype=np.float32)
            # cache a copy in demo for faster subsequent loads
            try:
                np.save(EMB_PATH, arr)
            except Exception:
                pass
            return arr
        except Exception:
            return None
    # try looking for any .npy files in the embed folder
    # fallback: search any candidate folder for .npy files
    for embed_folder in candidates:
        if os.path.isdir(embed_folder):
            files = [os.path.join(embed_folder, f) for f in os.listdir(embed_folder) if f.endswith('.npy')]
        if files:
            try:
                mats = [np.load(f) for f in sorted(files)]
                arr = np.vstack(mats)
                np.save(EMB_PATH, arr)
                return arr
            except Exception:
                return None
    return None


def load_index(expected_dim=None):
    """Load the Annoy index and return (index, dim).

    If `index_meta.json` exists, prefer the stored dimension; otherwise fall
    back to `expected_dim` if provided. This helps detect mismatches between
    a previously-built index and the embedding dimensionality produced at
    inference time.
    """
    if not os.path.exists(INDEX_PATH):
        return None, None

    # prefer metadata written when the index was built
    dim = None
    try:
        if os.path.exists(INDEX_META_PATH):
            with open(INDEX_META_PATH, 'r') as f:
                j = json.load(f)
                dim = int(j.get('dim')) if j.get('dim') is not None else None
    except Exception:
        dim = None

    if dim is None:
        # fall back to caller-provided expected_dim
        dim = expected_dim
    if dim is None:
        # last resort: try to load with a reasonable default (will likely fail later)
        logger.warning('load_index: no index metadata available and no expected_dim provided')
        return None, None

    t = AnnoyIndex(int(dim), 'angular')
    t.load(INDEX_PATH)
    return t, int(dim)


@app.route('/api/health', methods=['GET'])
def api_health():
    """Return basic health/readiness info for the demo: embeddings, index, embedder availability."""
    # check torch availability and embedder checkpoint
    torch_avail = False
    try:
        import importlib
        spec = importlib.util.find_spec('torch')
        torch_avail = spec is not None
    except Exception:
        torch_avail = False

    emb_ckpt_paths = [
        os.path.join(BASE_DIR, 'models', 'embedder_mirex', 'song_embedder.pt'),
        os.path.join(BASE_DIR, 'models', 'embedder_top40', 'song_embedder.pt'),
    ]
    ckpt_exists = any(os.path.exists(p) for p in emb_ckpt_paths)

    embeddings_present = load_embeddings() is not None
    index_built = os.path.exists(INDEX_PATH)
    return jsonify({
        'ok': True,
        'torch_available': bool(torch_avail),
        'embedder_checkpoint_present': bool(ckpt_exists),
        'embeddings_present': bool(embeddings_present),
        'index_built': bool(index_built),
    })


@app.route('/api/health/verbose', methods=['GET'])
def api_health_verbose():
    """Return verbose diagnostics about the embedder checkpoint and inferred architecture."""
    model_dirs = [
        os.path.join(BASE_DIR, 'models', 'embedder_mirex'),
        os.path.join(BASE_DIR, 'models', 'embedder_top40'),
    ]
    info = {
        'ckpt_checked': None,
        'ckpt_keys': None,
        'inferred_hidden_dims': None,
        'inferred_num_classes': None,
        'input_dim': None,
        'embedding_dim': None,
        'sample_feature_keys': None,
    }
    for d in model_dirs:
        ckpt_path = os.path.join(d, 'song_embedder.pt')
        if os.path.exists(ckpt_path):
            info['ckpt_checked'] = ckpt_path
            try:
                import torch as _t
                ckpt = _t.load(ckpt_path, map_location='cpu')
                info['ckpt_keys'] = list(ckpt.keys())
                info['input_dim'] = int(ckpt.get('input_dim', 0))
                info['embedding_dim'] = int(ckpt.get('embedding_dim', 0))
                info['inferred_num_classes'] = int(ckpt.get('num_classes')) if ckpt.get('num_classes') else None
                state_dict = ckpt.get('model_state_dict') or ckpt.get('model') or ckpt.get('model_state')
                if state_dict is not None:
                    # sample feature_extractor keys and shapes
                    feat_keys = [k for k in state_dict.keys() if k.startswith('feature_extractor.')]
                    info['sample_feature_keys'] = {k: (list(state_dict[k].shape) if hasattr(state_dict[k], 'shape') else str(type(state_dict[k]))) for k in feat_keys[:20]}
                    # infer hidden dims using same logic as loader
                    import re
                    linear_entries = []
                    for k in sorted([k for k in state_dict.keys() if k.startswith('feature_extractor.') and k.endswith('.weight')], key=lambda x: int(re.match(r'feature_extractor\.(\d+)\.weight', x).group(1))):
                        tensor = state_dict.get(k)
                        if tensor is None:
                            continue
                        if hasattr(tensor, 'ndim') and int(getattr(tensor, 'ndim', 0)) == 2:
                            idx = int(re.match(r'feature_extractor\.(\d+)\.weight', k).group(1))
                            outf = int(tensor.shape[0])
                            linear_entries.append((idx, outf))
                    hidden_dims = [out for (_, out) in sorted(linear_entries, key=lambda x: x[0])]
                    if hidden_dims:
                        info['inferred_hidden_dims'] = hidden_dims
            except Exception as e:
                info['error'] = str(e)
            break
    return jsonify({'ok': True, 'verbose': info})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Accept an audio file upload, compute embedding, and return nearest neighbors."""
    # validate file
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'no file uploaded (field name should be "file")'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'ok': False, 'error': 'empty filename'}), 400

    # save to temp
    tmp_dir = tempfile.mkdtemp(prefix='demo_upload_')
    tmp_path = os.path.join(tmp_dir, f.filename)
    try:
        f.save(tmp_path)

        # try to use repository extractor if available
        features = None
        try:
            sys.path.insert(0, os.path.join(BASE_DIR, 'utilities', 'src'))
            from extraction import extract
            y, sr = extract.load_audio(tmp_path)
            features = {}
            try:
                features.update(extract.perceptual_mfcc(y, sr))
            except Exception:
                pass
            try:
                features.update(extract.chroma_features(y, sr))
            except Exception:
                pass
            try:
                features.update(extract.spectral_descriptors(y, sr))
            except Exception:
                pass
            try:
                features.update(extract.rhythm_features(y, sr))
            except Exception:
                pass
        except Exception:
            # fallback to librosa-based minimal features
            # try lazy import of librosa/soundfile for fallback
            global _LIBROSA_AVAILABLE, librosa, sf
            if not _LIBROSA_AVAILABLE:
                try:
                    import importlib
                    if importlib.util.find_spec('librosa'):
                        import librosa as _lib
                        librosa = _lib
                        try:
                            import soundfile as _sf
                            sf = _sf
                        except Exception:
                            sf = None
                        _LIBROSA_AVAILABLE = True
                        logger.info('librosa imported for fallback extraction')
                except Exception:
                    logger.exception('failed to import librosa for fallback')
            if not _LIBROSA_AVAILABLE:
                return jsonify({'ok': False, 'error': 'neither repository extractor nor librosa available'}), 500
            try:
                y, sr = librosa.load(tmp_path, sr=22050, mono=True)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features = {
                    'mfcc_mean': list(np.mean(mfcc, axis=1)),
                    'mfcc_std': list(np.std(mfcc, axis=1)),
                }
            except Exception as e:
                logger.exception('librosa extraction failed: %s', e)
                return jsonify({'ok': False, 'error': f'feature extraction failed: {e}'}), 500

        # flatten features to vector
        flat = flatten_extracted_features(features)

        # load embedder (cached)
        global _CACHED_EMBEDDER
        if _CACHED_EMBEDDER is None:
            _CACHED_EMBEDDER = load_embedder_checkpoint()
        if not _CACHED_EMBEDDER or _CACHED_EMBEDDER.get('model') is None:
            return jsonify({'ok': False, 'error': 'embedder checkpoint not available or torch not installed'}), 500

        model = _CACHED_EMBEDDER['model']
        scaler = _CACHED_EMBEDDER.get('scaler')
        feature_cols = _CACHED_EMBEDDER.get('feature_cols', [])

        # build input vector respecting feature_cols order if available
        if feature_cols:
            vec = [float(flat.get(c, 0.0)) for c in feature_cols]
        else:
            # fallback: take sorted keys
            keys = sorted(flat.keys())
            vec = [float(flat[k]) for k in keys]
            feature_cols = keys

        x = np.array(vec, dtype=np.float32).reshape(1, -1)
        # apply scaler if present
        if scaler is not None:
            try:
                x = scaler.transform(x)
            except Exception:
                pass

        # ensure input dims match model expected input; pad or trim
        expected = None
        try:
            # try to inspect the feature_extractor first linear layer
            if hasattr(model, 'feature_extractor') and len(getattr(model, 'feature_extractor')) > 0:
                first = model.feature_extractor[0]
                if hasattr(first, 'in_features'):
                    expected = int(first.in_features)
        except Exception:
            expected = None
        if expected is None:
            # fall back to checkpoint-declared input_dim if available
            expected = int(_CACHED_EMBEDDER.get('input_dim', x.shape[1]))
        if x.shape[1] < expected:
            pad = np.zeros((1, expected - x.shape[1]), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=1)
        elif x.shape[1] > expected:
            # trim extra features (log for debugging)
            logger.warning('input feature length (%d) > model expected (%d); trimming extra features', x.shape[1], expected)
            x = x[:, :expected]

        # run model
        try:
            import torch as _torch
            _inp = _torch.from_numpy(x).float()
            with _torch.no_grad():
                out = model(_inp)
                # model returns (embedding, logits) during training; accept both
                if isinstance(out, tuple) or isinstance(out, list):
                    emb = out[0].cpu().numpy().reshape(-1)
                else:
                    emb = out.cpu().numpy().reshape(-1)
        except Exception as e:
            return jsonify({'ok': False, 'error': f'model inference failed: {e}'}), 500

        # query index
        embeddings = load_embeddings()
        if embeddings is None:
            return jsonify({'ok': False, 'error': 'embeddings not found (build index first)'}), 400
        # load index and its stored dimension (index_meta.json)
        index, index_dim = load_index()
        if index is None or index_dim is None:
            return jsonify({'ok': False, 'error': 'index not built yet (or index metadata missing)'}), 400

        k = int(request.form.get('k', 10))
        # verify embedding length matches index dimension
        if len(emb) != int(index_dim):
            return jsonify({'ok': False, 'error': f'embedding dimension mismatch: index expects {index_dim}, got {len(emb)}. Rebuild the index with the same embedder.'}), 400
        neighbors, distances = index.get_nns_by_vector(emb.tolist(), k, include_distances=True)
        meta = load_metadata()
        results = []
        for n, d in zip(neighbors, distances):
            info = meta.get(str(n), {}) if isinstance(meta, dict) else {}
            results.append({'id': int(n), 'distance': float(d), 'info': info})

        return jsonify({'ok': True, 'embedding': emb.tolist(), 'results': results})
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


# ------------------ Inference helpers ------------------
def find_feature_columns():
    """Attempt to discover the original numeric feature column order used for training.

    This reads a features parquet/csv header from the repo (features/mirex) and
    returns a list of numeric column names.
    """
    candidates = [
        os.path.join(BASE_DIR, 'features', 'mirex', 'features_flat_clean_clean.parquet'),
        os.path.join(BASE_DIR, 'features', 'mirex', 'features_flat_clean.parquet'),
        os.path.join(BASE_DIR, 'features', 'mirex', 'features_flat.parquet'),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                import pandas as pd
                df = pd.read_parquet(p)
                # exclude metadata-like columns
                exclude = {'track_id', 'mood_cluster', 'mood_category', 'dataset', 'language'}
                cols = [c for c in df.columns if c not in exclude]
                # keep only numeric dtype columns
                num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
                return num_cols
            except Exception:
                continue
    return []


def flatten_extracted_features(feat: dict):
    """Flatten the nested features dict produced by `utilities/src/extraction/extract.py`.

    This attempts to produce keys matching the flattened feature table such as
    'mfcc_0_mean', 'mfcc_0_std', 'spectral_centroid_mean', etc.
    Missing keys will be omitted.
    """
    flat = {}

    def _to_number(v):
        """Convert scalar or list/ndarray to a single float safely.

        For lists/ndarrays returns the mean of numeric entries. Raises ValueError
        for non-numeric content.
        """
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v, dtype=np.float64).ravel()
                # keep finite numbers
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    raise ValueError('no numeric values')
                return float(np.mean(arr))
            except Exception:
                raise ValueError('non-numeric list')
        raise ValueError('not a number')

    def _visit(prefix, obj):
        if obj is None:
            return
        if isinstance(obj, (int, float, np.floating, np.integer)):
            flat[prefix] = float(obj)
            return
        if isinstance(obj, list) or isinstance(obj, np.ndarray):
            # heuristics for common list-valued features
            if prefix in ('mfcc_mean', 'mfcc_std') or prefix.startswith('mfcc_'):
                # expand to mfcc_0_mean, mfcc_1_mean, ...
                for i, v in enumerate(obj):
                    key = f'mfcc_{i}_' + ('mean' if 'mean' in prefix else 'std' if 'std' in prefix else 'val')
                    try:
                        flat[key] = _to_number(v)
                    except Exception:
                        # skip non-numeric entries
                        continue
                return
            if prefix in ('chroma_mean', 'chroma_std'):
                for i, v in enumerate(obj):
                    key = f'chroma_{i}_' + ('mean' if 'mean' in prefix else 'std' if 'std' in prefix else 'val')
                    try:
                        flat[key] = _to_number(v)
                    except Exception:
                        continue
                return
            # generic list: index keys
            for i, v in enumerate(obj):
                _visit(f"{prefix}_{i}", v)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                # special-case some names produced by extractor
                if k == 'mfcc_mean':
                    _visit('mfcc_mean', v)
                elif k == 'mfcc_std':
                    _visit('mfcc_std', v)
                elif k == 'chroma_mean':
                    _visit('chroma_mean', v)
                elif k == 'chroma_std':
                    _visit('chroma_std', v)
                else:
                    _visit(k, v)
            return
        # fallback: try to coerce to a numeric scalar
        try:
            flat[prefix] = _to_number(obj)
        except Exception:
            pass

    _visit('', feat)
    # clean keys: remove leading underscore if present
    cleaned = {k.lstrip('_'): v for k, v in flat.items() if k}
    return cleaned


def load_embedder_checkpoint(prefer='embedder_mirex'):
    """Load song_embedder.pt checkpoint and return a small inference struct.

    Returns dict with keys: model, scaler, feature_cols, embedding_dim
    """
    # try to import torch lazily if not available yet
    global _TORCH_AVAILABLE, torch
    if not _TORCH_AVAILABLE:
        try:
            import importlib
            torch_spec = importlib.util.find_spec('torch')
            if torch_spec is not None:
                import torch as _t
                torch = _t
                _TORCH_AVAILABLE = True
                logger.info('torch available for embedder loading')
            else:
                logger.warning('torch not available')
                return None
        except Exception:
            logger.exception('failed to import torch')
            return None
    model_dirs = {
        'embedder_mirex': os.path.join(BASE_DIR, 'models', 'embedder_mirex'),
        'embedder_top40': os.path.join(BASE_DIR, 'models', 'embedder_top40'),
    }
    d = model_dirs.get(prefer, model_dirs['embedder_mirex'])
    ckpt_path = os.path.join(d, 'song_embedder.pt')
    if not os.path.exists(ckpt_path):
        return None
    try:
        logger.info('loading embedder checkpoint from %s', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        input_dim = int(ckpt.get('input_dim', 0))
        embedding_dim = int(ckpt.get('embedding_dim', ckpt.get('embedding_dim', 64)))
        # examine state_dict to infer hidden layer sizes used during training
        state_dict = ckpt.get('model_state_dict') or ckpt.get('model') or ckpt.get('model_state')
        inferred_hidden_dims = None
        inferred_num_classes = int(ckpt.get('num_classes', 0)) if ckpt.get('num_classes') else None
        if state_dict is not None:
            try:
                import re
                # Only consider linear layer weights (2-D tensors) to infer hidden dims
                feat_keys = [k for k in state_dict.keys() if k.startswith('feature_extractor.') and k.endswith('.weight')]
                linear_entries = []
                for k in sorted(feat_keys, key=lambda x: int(re.match(r'feature_extractor\.(\d+)\.weight', x).group(1))):
                    tensor = state_dict.get(k)
                    if tensor is None:
                        continue
                    # linear layers have 2D weight tensors (out_features, in_features)
                    if hasattr(tensor, 'ndim') and int(getattr(tensor, 'ndim', 0)) == 2:
                        idx = int(re.match(r'feature_extractor\.(\d+)\.weight', k).group(1))
                        outf = int(tensor.shape[0])
                        linear_entries.append((idx, outf))
                # Remove duplicates and keep order, only use output sizes
                linear_entries = sorted(linear_entries, key=lambda x: x[0])
                hidden_dims = [out for (_, out) in linear_entries]
                if hidden_dims:
                    inferred_hidden_dims = tuple(hidden_dims)
                # infer num_classes from classifier weight if not provided
                if inferred_num_classes is None and 'classifier.weight' in state_dict:
                    inferred_num_classes = int(state_dict['classifier.weight'].shape[0])
            except Exception:
                inferred_hidden_dims = None
        # fallback defaults
        if inferred_hidden_dims is None:
            inferred_hidden_dims = (256, 128)
        if inferred_num_classes is None:
            inferred_num_classes = int(ckpt.get('num_classes', 2) or 2)
        # Reconstruct model architecture matching training `SongEmbedder`
        class SongEmbedderSimple(torch.nn.Module):
            def __init__(self, input_dim, embedding_dim, num_classes=2, hidden_dims=inferred_hidden_dims, dropout=0.1):
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden_dims:
                    layers.append(torch.nn.Linear(prev, h))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.BatchNorm1d(h))
                    layers.append(torch.nn.Dropout(dropout))
                    prev = h
                self.feature_extractor = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
                self.embedding_layer = torch.nn.Linear(prev, embedding_dim)
                self.classifier = torch.nn.Linear(embedding_dim, int(ckpt.get('num_classes', num_classes)))

            def forward(self, x):
                h = self.feature_extractor(x)
                emb = self.embedding_layer(h)
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                logits = self.classifier(emb)
                return emb, logits

        model = SongEmbedderSimple(input_dim, embedding_dim, num_classes=int(ckpt.get('num_classes', 2)))
        # support multiple possible keys used when saving checkpoints
        state_dict = ckpt.get('model_state_dict') or ckpt.get('model') or ckpt.get('model_state')
        if state_dict is None:
            logger.error('no model_state found in checkpoint keys: %s', list(ckpt.keys()))
            return None
        model.load_state_dict(state_dict)
        model.eval()

        # scaler reconstruction
        scaler = None
        if 'scaler_mean' in ckpt and 'scaler_scale' in ckpt:
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.mean_ = np.array(ckpt['scaler_mean'], dtype=np.float64)
                scaler.scale_ = np.array(ckpt['scaler_scale'], dtype=np.float64)
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_samples_seen_ = np.array([1])
                logger.info('reconstructed StandardScaler from checkpoint')
            except Exception as e:
                logger.exception('failed to reconstruct scaler: %s', e)
                scaler = None

        # try to infer feature columns from features file
        feature_cols = find_feature_columns()

        return {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'embedding_dim': embedding_dim,
            'input_dim': input_dim,
        }
    except Exception:
        logger.exception('failed to load embedder checkpoint')
        return None


_CACHED_EMBEDDER = None



@app.route('/')
def index():
    meta = load_metadata()
    return render_template('index.html', metadata=meta)


@app.route('/static_demo')
def static_demo():
    # look for plots inside analysis and model embedder
    plots = []
    search_dirs = [
        os.path.join(BASE_DIR, 'models', 'embedder_mirex', 'plots'),
        os.path.join(BASE_DIR, 'analysis', 'mirex_full', 'plots'),
        os.path.join(BASE_DIR, 'Analysis', 'jamando', 'plots'),
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                        rel = os.path.relpath(os.path.join(root, f), BASE_DIR)
                        plots.append(rel.replace('\\', '/'))
    return render_template('static_demo.html', plots=plots)


@app.route('/api/index', methods=['POST'])
def api_index():
    # Build or rebuild index from embeddings
    embeddings = load_embeddings()
    if embeddings is None:
        return jsonify({'ok': False, 'error': 'No embeddings found (looked in models/embedder_mirex/embeddings)'}), 400
    dim = embeddings.shape[1]
    t = AnnoyIndex(dim, 'angular')
    for i, vec in enumerate(embeddings):
        t.add_item(i, vec.astype(np.float32))
    t.build(10)
    t.save(INDEX_PATH)
    # write index metadata so callers can verify expected dimensionality
    try:
        with open(INDEX_META_PATH, 'w') as f:
            json.dump({'dim': int(dim), 'items': int(embeddings.shape[0])}, f)
    except Exception:
        logger.exception('failed to write index metadata')
    # save embeddings copy
    np.save(EMB_PATH, embeddings)
    # copy metadata if available
    meta = load_metadata()
    with open(META_PATH, 'w') as f:
        json.dump(meta, f)
    return jsonify({'ok': True, 'items_indexed': int(embeddings.shape[0])})


@app.route('/api/library', methods=['GET'])
def api_library():
    meta = load_metadata()
    return jsonify({'ok': True, 'metadata': meta})


@app.route('/api/search', methods=['POST'])
def api_search():
    payload = request.json or {}
    k = int(payload.get('k', 10))
    song_id = payload.get('song_id')
    embeddings = load_embeddings()
    if embeddings is None:
        return jsonify({'ok': False, 'error': 'embeddings not found'}), 400
    # load index and its stored dimension (index_meta.json)
    index, index_dim = load_index()
    if index is None or index_dim is None:
        return jsonify({'ok': False, 'error': 'index not built yet (or index metadata missing)'}), 400
    if song_id is not None:
        try:
            idx = int(song_id)
            neighbors, distances = index.get_nns_by_item(idx, k, include_distances=True)
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 400
    else:
        emb = payload.get('embedding')
        if emb is None:
            return jsonify({'ok': False, 'error': 'provide song_id or embedding'}), 400
        emb = np.array(emb, dtype=np.float32)
        # validate embedding size matches index
        if emb.ndim == 1 and int(emb.size) != int(index_dim):
            return jsonify({'ok': False, 'error': f'embedding dimension mismatch: index expects {index_dim}, got {emb.size}. Rebuild the index with the same embedder.'}), 400
        neighbors, distances = index.get_nns_by_vector(emb, k, include_distances=True)
    meta = load_metadata()
    results = []
    for n, d in zip(neighbors, distances):
        info = meta.get(str(n), {}) if isinstance(meta, dict) else {}
        results.append({'id': int(n), 'distance': float(d), 'info': info})
    return jsonify({'ok': True, 'results': results})


@app.route('/file/<path:filepath>')
def serve_file(filepath):
    # Serve arbitrary repo file (use carefully in demo only)
    safe_base = BASE_DIR
    full = os.path.abspath(os.path.join(safe_base, filepath))
    if not full.startswith(safe_base):
        return 'Forbidden', 403
    directory = os.path.dirname(full)
    filename = os.path.basename(full)
    if not os.path.exists(full):
        return 'Not found', 404
    return send_from_directory(directory, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
