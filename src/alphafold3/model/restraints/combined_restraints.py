"""CombinedRestraints for AlphaFold3: manages conformer and distance restraints.

Differences from Protenix version:
  - Positions shape: (num_tokens, max_atoms_per_token, 3) instead of (N_batch, N_atom_flat, 3)
  - active_sites: flat indices (token_idx * max_atoms_per_token + atom_in_token_idx)
  - GPU optimizer: jax.scipy.optimize.minimize(BFGS) instead of torchmin
  - CPU optimizer: scipy.optimize.minimize(CG) - same as Protenix
  - minimize() takes single-sample positions (3D array) instead of batched
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from rdkit import Chem
from scipy import optimize

from .chiral_data import ChiralData, calc_chiral_vol
from .angle_restr_data import AngleData, get_angle_idxs
from .bond_restr_data import BondData
from .distance_restr_data import DistanceData


class CombinedRestraints:
    """Manages all restraints applied during diffusion sampling for AlphaFold3.

    Singleton pattern: call CombinedRestraints.get_instance() to obtain the shared instance.
    Reset with CombinedRestraints._instance = None before each new prediction job.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> CombinedRestraints:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.chiral_data: list[ChiralData] = []
        self.bond_data: list[BondData] = []
        self.angle_data: list[AngleData] = []
        self.distance_data: list[DistanceData] = []
        # sites[sid-1]: list of setup callbacks for site sid (1-indexed; 0 = unconstrained)
        self.sites: list[list] = []
        # Maps global flat atom index -> site_id (1-indexed)
        self.site_registry: dict[int, int] = {}
        # Set after setup_site(); flat atom global indices included in optimization
        self.active_sites: list[int] = []
        # Config
        self.config: dict = {}
        self.verbose: bool = False
        self.method: str = "CG"
        self.max_iter: int = 100
        self.start_sigma: float = 1.0
        self.gpu: bool = False
        self.bond_config: dict = {}
        self.angle_config: dict = {}
        self.chiral_config: dict = {}
        # State for scipy minimize (single-sample path)
        self.natoms: int = 0
        # Precomputed JAX arrays for GPU path (set in _setup_jax)
        self._jax_arrays: dict = {}
        # Shape info set in setup_site
        self._max_atoms_per_token: int = 1

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_config(self, config: dict) -> None:
        self.config = config
        self.verbose = config.get("verbose", False)
        self.method = config.get("method", "CG")
        self.max_iter = int(config.get("max_iter", 100))
        self.start_sigma = float(config.get("start_sigma", 1.0))
        self.gpu = config.get("gpu", False)

        conformer_cfg = config.get("conformer_restraints_config", {})
        self.bond_config = conformer_cfg.get("bond", {})
        self.angle_config = conformer_cfg.get("angle", {})
        self.chiral_config = conformer_cfg.get("chiral", {})

        distance_cfg = config.get("distance_restraints_config", [])
        for entry in distance_cfg:
            dr = DistanceData()
            dr.set_config(entry)
            self.distance_data.append(dr)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def _create_bond_data(self, d: float) -> BondData:
        return BondData(
            -1, -1, d,
            slack=self.bond_config.get("slack", 0.0),
            w=self.bond_config.get("weight", 0.05),
        )

    def _create_angle_data(self, th0: float) -> AngleData:
        return AngleData(
            -1, -1, -1, th0,
            slack=self.angle_config.get("slack", math.radians(5.0)),
            w=self.angle_config.get("weight", 0.05),
        )

    def _create_chiral_data(self, chiral_vol: float) -> ChiralData:
        return ChiralData(
            -1, -1, -1, -1, chiral_vol,
            slack=self.chiral_config.get("slack", 0.05),
            w=self.chiral_config.get("weight", 0.05),
            fmax=self.chiral_config.get("f_max", -100.0),
        )

    # ------------------------------------------------------------------
    # Site registry
    # ------------------------------------------------------------------

    def register_site(self, flat_global_idx: int, callback) -> None:
        """Register a global flat atom index with a setup callback."""
        sid = self.site_registry.get(flat_global_idx, 0)
        if sid == 0:
            self.sites.append([callback])
            new_sid = len(self.sites)
            self.site_registry[flat_global_idx] = new_sid
        else:
            self.sites[sid - 1].append(callback)

    def get_sites(self, index: int) -> list:
        if index == 0:
            return []
        return self.sites[index - 1]

    # ------------------------------------------------------------------
    # Conformer restraint builders
    # ------------------------------------------------------------------

    def make_bond(self, rdkit_ai: int, rdkit_aj: int, conf, global_indices: np.ndarray) -> None:
        """Register a bond length restraint.

        Args:
            rdkit_ai, rdkit_aj: atom indices in H-removed RDKit mol.
            conf: RDKit conformer of H-removed mol.
            global_indices: array mapping mol atom idx -> flat global index.
        """
        crds = conf.GetPositions()
        d = np.linalg.norm(crds[rdkit_aj] - crds[rdkit_ai])
        bnd = self._create_bond_data(d)
        self.bond_data.append(bnd)
        global_ai = int(global_indices[rdkit_ai])
        global_aj = int(global_indices[rdkit_aj])
        self.register_site(global_ai, lambda x, b=bnd: b.setup(x, 0))
        self.register_site(global_aj, lambda x, b=bnd: b.setup(x, 1))

    def make_angle(self, rdkit_ai: int, rdkit_aj: int, rdkit_ak: int, conf, global_indices) -> None:
        """Register a bond angle restraint."""
        th0 = AngleData.calc_angle(rdkit_ai, rdkit_aj, rdkit_ak, conf)
        angl = self._create_angle_data(th0)
        self.angle_data.append(angl)
        global_ai = int(global_indices[rdkit_ai])
        global_aj = int(global_indices[rdkit_aj])
        global_ak = int(global_indices[rdkit_ak])
        self.register_site(global_ai, lambda x, a=angl: a.setup(x, 0))
        self.register_site(global_aj, lambda x, a=angl: a.setup(x, 1))
        self.register_site(global_ak, lambda x, a=angl: a.setup(x, 2))

    def make_angle_restraints(self, mol: Chem.Mol, conf, global_indices) -> None:
        """Register angle restraints for all 1-2-3 triples in mol."""
        for ai, aj, ak in get_angle_idxs(mol):
            self.make_angle(int(ai), int(aj), int(ak), conf, global_indices)

    def make_chiral_impl(self, rdkit_ai: int, rdkit_aj: list[int], conf, global_indices, invert: bool = False) -> None:
        crds = conf.GetPositions()
        chiral_vol = calc_chiral_vol(crds, rdkit_ai, rdkit_aj)
        if invert:
            chiral_vol = -chiral_vol
        ch = self._create_chiral_data(chiral_vol)
        self.chiral_data.append(ch)
        global_ai = int(global_indices[rdkit_ai])
        global_ajs = [int(global_indices[j]) for j in rdkit_aj]
        self.register_site(global_ai, lambda x, c=ch: c.setup(x, 0))
        self.register_site(global_ajs[0], lambda x, c=ch: c.setup(x, 1))
        self.register_site(global_ajs[1], lambda x, c=ch: c.setup(x, 2))
        self.register_site(global_ajs[2], lambda x, c=ch: c.setup(x, 3))
        if self.verbose:
            print(f"chiral restr {rdkit_ai} - {rdkit_aj}: vol={chiral_vol:.2f}")

    def make_chiral(self, iatm: int, mol: Chem.Mol, conf, global_indices, invert: bool = False) -> None:
        nei_ind = ChiralData.get_nei_atoms(iatm, mol)
        for cand in itertools.combinations(nei_ind, 3):
            self.make_chiral_impl(iatm, list(cand), conf, global_indices, invert=invert)

    # ------------------------------------------------------------------
    # Feature array builder
    # ------------------------------------------------------------------

    def build_conformer_restraint_array(self, N_flat: int) -> np.ndarray:
        """Build ref_conformer_restraint array for AlphaFold3.

        Args:
            N_flat: total flat atom count = num_tokens * max_atoms_per_token

        Returns:
            int64 array of shape (N_flat,): 0 = no restraint, >0 = site_id.
        """
        arr = np.zeros(N_flat, dtype=np.int64)
        for flat_idx, site_id in self.site_registry.items():
            if flat_idx < N_flat:
                arr[flat_idx] = site_id
        return arr

    # ------------------------------------------------------------------
    # Distance restraint setup
    # ------------------------------------------------------------------

    def set_feats_af3(self, all_token_atoms_layout, max_atoms_per_token: int) -> None:
        """Resolve distance restraint selections to flat global indices."""
        for dr in self.distance_data:
            dr.set_feats_af3(all_token_atoms_layout, max_atoms_per_token)

    # ------------------------------------------------------------------
    # Setup (called before diffusion sampling)
    # ------------------------------------------------------------------

    def setup_site(
        self,
        conformer_restraint_arr: np.ndarray | None,
        max_atoms_per_token: int,
    ) -> None:
        """Resolve site indices and build active_sites list.

        Args:
            conformer_restraint_arr: shape (N_flat,) int64 array from
                build_conformer_restraint_array(), or None if no conformer restraints.
            max_atoms_per_token: needed to convert flat idx to 2D for JAX.
        """
        self.reset_indices()
        self._max_atoms_per_token = max_atoms_per_token

        # Reset local site lists for distance restraints
        for dr in self.distance_data:
            dr.target_local_sites1 = []
            dr.target_local_sites2 = []

        self.active_sites = []

        # Add atoms from conformer restraints
        if conformer_restraint_arr is not None:
            for ind in range(len(conformer_restraint_arr)):
                if int(conformer_restraint_arr[ind]) != 0:
                    self.active_sites.append(ind)

        # Add atoms from distance restraints
        for dr in self.distance_data:
            if dr.run_restr:
                self.active_sites += dr.target_sites1
                self.active_sites += dr.target_sites2

        if len(self.active_sites) == 0:
            return

        self.active_sites = sorted(set(self.active_sites))
        self.natoms = len(self.active_sites)

        if self.verbose:
            print(f"[CombinedRestraints] active_sites={self.natoms} atoms")

        # Build local index mappings
        for local_i, global_idx in enumerate(self.active_sites):
            # Map distance restraint global -> local
            for dr in self.distance_data:
                if global_idx in set(dr.target_sites1):
                    dr.target_local_sites1.append(local_i)
                if global_idx in set(dr.target_sites2):
                    dr.target_local_sites2.append(local_i)

            # Resolve conformer restraint callbacks
            if conformer_restraint_arr is not None and global_idx < len(conformer_restraint_arr):
                sid = int(conformer_restraint_arr[global_idx])
                if sid != 0:
                    for callback in self.get_sites(sid):
                        callback(local_i)

        if self.verbose:
            for ch in self.chiral_data:
                if ch.is_valid():
                    print(f"  chiral: {ch.aid0}-{ch.aid1}-{ch.aid2}-{ch.aid3}")

        print(
            f"[CombinedRestraints] active_sites={self.natoms},"
            f" bonds={len(self.bond_data)}, angles={len(self.angle_data)},"
            f" chirals={len(self.chiral_data)}, distances={len(self.distance_data)}"
        )

        if self.gpu:
            self._setup_jax()

    def is_active(self) -> bool:
        return len(self.active_sites) > 0

    # ------------------------------------------------------------------
    # JAX GPU setup
    # ------------------------------------------------------------------

    def _setup_jax(self) -> None:
        """Precompute JAX arrays for GPU-based BFGS minimization."""
        import jax.numpy as jnp

        # Bond arrays
        valid_bonds = [b for b in self.bond_data if b.is_valid()]
        if valid_bonds:
            self._jax_arrays['bond_aid0'] = jnp.array([b.aid0 for b in valid_bonds])
            self._jax_arrays['bond_aid1'] = jnp.array([b.aid1 for b in valid_bonds])
            self._jax_arrays['bond_r0'] = jnp.array([b.r0 for b in valid_bonds])
            self._jax_arrays['bond_slack'] = jnp.array([b.slack for b in valid_bonds])
            self._jax_arrays['bond_w'] = jnp.array([b.w for b in valid_bonds])

        # Angle arrays
        valid_angles = [a for a in self.angle_data if a.is_valid()]
        if valid_angles:
            self._jax_arrays['angle_aid0'] = jnp.array([a.aid0 for a in valid_angles])
            self._jax_arrays['angle_aid1'] = jnp.array([a.aid1 for a in valid_angles])
            self._jax_arrays['angle_aid2'] = jnp.array([a.aid2 for a in valid_angles])
            self._jax_arrays['angle_th0'] = jnp.array([a.th0 for a in valid_angles])
            self._jax_arrays['angle_slack'] = jnp.array([a.slack for a in valid_angles])
            self._jax_arrays['angle_w'] = jnp.array([a.w for a in valid_angles])

        # Chiral arrays
        valid_chirals = [c for c in self.chiral_data if c.is_valid()]
        if valid_chirals:
            self._jax_arrays['chiral_aid0'] = jnp.array([c.aid0 for c in valid_chirals])
            self._jax_arrays['chiral_aid1'] = jnp.array([c.aid1 for c in valid_chirals])
            self._jax_arrays['chiral_aid2'] = jnp.array([c.aid2 for c in valid_chirals])
            self._jax_arrays['chiral_aid3'] = jnp.array([c.aid3 for c in valid_chirals])
            self._jax_arrays['chiral_vol'] = jnp.array([c.chiral_vol for c in valid_chirals])
            self._jax_arrays['chiral_slack'] = jnp.array([c.slack for c in valid_chirals])
            self._jax_arrays['chiral_w'] = jnp.array([c.w for c in valid_chirals])

        # Distance arrays (per DistanceData)
        self._jax_arrays['distance_data'] = [
            {
                'sites1': jnp.array(dr.target_local_sites1),
                'sites2': jnp.array(dr.target_local_sites2),
                'target_distance': dr.target_distance,
                'target_distance1': dr.target_distance1,
                'target_distance2': dr.target_distance2,
                'rtype': dr.distance_restraint_type,
            }
            for dr in self.distance_data if dr.is_valid()
        ]

    def _jax_total_energy(self, positions_flat: 'jnp.ndarray') -> 'jnp.ndarray':
        """Total energy as a JAX-traceable function. positions_flat: (N_active*3,)"""
        from .jax_energy import (
            jax_bond_energy, jax_angle_energy, jax_chiral_energy, jax_distance_energy
        )
        import jax.numpy as jnp

        pos = positions_flat.reshape(-1, 3)  # (N_active, 3)
        ene = jnp.zeros(())

        if 'bond_aid0' in self._jax_arrays:
            ene += jax_bond_energy(
                pos,
                self._jax_arrays['bond_aid0'],
                self._jax_arrays['bond_aid1'],
                self._jax_arrays['bond_r0'],
                self._jax_arrays['bond_slack'],
                self._jax_arrays['bond_w'],
            )

        if 'angle_aid0' in self._jax_arrays:
            ene += jax_angle_energy(
                pos,
                self._jax_arrays['angle_aid0'],
                self._jax_arrays['angle_aid1'],
                self._jax_arrays['angle_aid2'],
                self._jax_arrays['angle_th0'],
                self._jax_arrays['angle_slack'],
                self._jax_arrays['angle_w'],
            )

        if 'chiral_aid0' in self._jax_arrays:
            ene += jax_chiral_energy(
                pos,
                self._jax_arrays['chiral_aid0'],
                self._jax_arrays['chiral_aid1'],
                self._jax_arrays['chiral_aid2'],
                self._jax_arrays['chiral_aid3'],
                self._jax_arrays['chiral_vol'],
                self._jax_arrays['chiral_slack'],
                self._jax_arrays['chiral_w'],
            )

        for dd in self._jax_arrays.get('distance_data', []):
            ene += jax_distance_energy(
                pos,
                dd['sites1'], dd['sites2'],
                dd['target_distance'], dd['target_distance1'], dd['target_distance2'],
                dd['rtype'],
            )

        return ene

    # ------------------------------------------------------------------
    # Minimization (single-sample, for pure_callback)
    # ------------------------------------------------------------------

    def minimize_single_sample(
        self, positions: np.ndarray, sigma_t: float
    ) -> np.ndarray:
        """Apply restraints to a single sample's positions.

        Args:
            positions: (num_tokens, max_atoms_per_token, 3) numpy array
            sigma_t: current noise level (float)

        Returns:
            Minimized positions with same shape.
        """
        if not self.is_active():
            return positions
        if sigma_t > self.start_sigma:
            return positions

        original_shape = positions.shape
        max_atoms = self._max_atoms_per_token
        N_flat = original_shape[0] * max_atoms

        # Flatten to (N_flat, 3) then slice active atoms
        # Convert to numpy to ensure mutability (positions may be a JAX DeviceArray)
        pos_flat = np.array(positions.reshape(N_flat, 3))
        active_sites_arr = np.array(self.active_sites)
        active_pos = pos_flat[active_sites_arr]  # (N_active, 3)

        if self.gpu:
            active_pos_opt = self._minimize_jax(active_pos)
        else:
            active_pos_opt = self._minimize_scipy(active_pos)

        pos_flat[active_sites_arr] = active_pos_opt
        return pos_flat.reshape(original_shape)

    def _minimize_scipy(self, active_pos: np.ndarray) -> np.ndarray:
        """CPU minimization via scipy CG."""
        self.natoms = len(self.active_sites)
        x0 = active_pos.reshape(-1)
        opt = optimize.minimize(
            self._calc, x0, jac=self._grad,
            method=self.method,
            options={"maxiter": self.max_iter},
        )
        if self.verbose:
            crds = opt.x.reshape(self.natoms, 3)
            for d in self.distance_data:
                if d.is_valid():
                    d.print(crds)
        return opt.x.reshape(self.natoms, 3)

    def _minimize_jax(self, active_pos: np.ndarray) -> np.ndarray:
        """GPU minimization via jax.scipy.optimize.minimize (BFGS).

        Runs in float64 to avoid NaN accumulation in the BFGS Hessian inverse
        approximation.  JAX disables float64 by default (x64 mode off), so we
        enable it only for this call via jax.experimental.enable_x64().

        Safe norms in jax_energy prevent NaN gradients at zero-length vectors.
        """
        import jax
        import jax.numpy as jnp
        import jax.scipy.optimize  # noqa: F401  explicit import required for JAX lazy loader

        def _run_bfgs():
            # float64 for numerical stability in BFGS Hessian approximation
            x0 = jnp.array(active_pos.reshape(-1), dtype=jnp.float64)

            def energy_f64(x):
                # Cast positions to float32 for jax_energy kernels, return float64
                return self._jax_total_energy(x.astype(jnp.float32)).astype(jnp.float64)

            return jax.scipy.optimize.minimize(
                energy_f64,
                x0,
                method='BFGS',
                options={'maxiter': self.max_iter, 'gtol': 1e-5},
            )

        try:
            with jax.experimental.enable_x64():
                result = _run_bfgs()
        except Exception:
            # Fallback: run without x64 if the context manager is unavailable
            result = jax.scipy.optimize.minimize(
                self._jax_total_energy,
                jnp.array(active_pos.reshape(-1), dtype=jnp.float32),
                method='BFGS',
                options={'maxiter': self.max_iter, 'gtol': 1e-5},
            )

        if self.verbose:
            fun_val = float(result.fun)
            status = "success" if result.success else f"failed (nit={result.nit})"
            if not np.isnan(fun_val):
                print(f"[Restraints JAX BFGS] {status}, fun={fun_val:.4f}")
            else:
                print(f"[Restraints JAX BFGS] {status}, fun=NaN (returning unmodified positions)")

        opt_x = np.array(result.x, dtype=np.float32)
        # Return unmodified positions if optimization produced NaN
        if np.any(np.isnan(opt_x)):
            return active_pos
        return opt_x.reshape(self.natoms, 3)

    # ------------------------------------------------------------------
    # make_restraint_callback - for use with jax.pure_callback
    # ------------------------------------------------------------------

    def make_restraint_callback(self):
        """Return a callback function for jax.pure_callback.

        The callback signature: (positions, t_hat) -> positions
        where positions is (num_tokens, max_atoms_per_token, 3) numpy float32.
        """
        def _callback(positions: np.ndarray, t_hat: np.ndarray) -> np.ndarray:
            sigma_t = float(t_hat)
            return self.minimize_single_sample(positions, sigma_t).astype(np.float32)
        return _callback

    # ------------------------------------------------------------------
    # Energy and gradient for scipy CG
    # ------------------------------------------------------------------

    def _calc(self, crds_flat: np.ndarray) -> float:
        crds = crds_flat.reshape(self.natoms, 3)
        ene = 0.0
        for ch in self.chiral_data:
            if ch.is_valid():
                ene += ch.calc(crds)
        for b in self.bond_data:
            if b.is_valid():
                ene += b.calc(crds)
        for a in self.angle_data:
            if a.is_valid():
                ene += a.calc(crds)
        for d in self.distance_data:
            if d.is_valid():
                ene += d.calc(crds)
        return ene

    def _grad(self, crds_flat: np.ndarray) -> np.ndarray:
        crds = crds_flat.reshape(self.natoms, 3)
        grad = np.zeros_like(crds)
        for ch in self.chiral_data:
            if ch.is_valid():
                ch.grad(crds, grad)
        for b in self.bond_data:
            if b.is_valid():
                b.grad(crds, grad)
        for a in self.angle_data:
            if a.is_valid():
                a.grad(crds, grad)
        for d in self.distance_data:
            if d.is_valid():
                d.grad(crds, grad)
        return grad.reshape(-1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_indices(self) -> None:
        for ch in self.chiral_data:
            ch.reset_indices()
        for b in self.bond_data:
            b.reset_indices()
        for a in self.angle_data:
            a.reset_indices()

    def print_stat(self, active_pos: np.ndarray) -> None:
        if self.chiral_data:
            ch_ene = sum(c.calc(active_pos) for c in self.chiral_data if c.is_valid())
            print(f"  chiral E={ch_ene:.5f}")
        if self.bond_data:
            b_ene = sum(b.calc(active_pos) for b in self.bond_data if b.is_valid())
            print(f"  bond   E={b_ene:.5f}")
        if self.angle_data:
            a_ene = sum(a.calc(active_pos) for a in self.angle_data if a.is_valid())
            print(f"  angle  E={a_ene:.5f}")
        if self.distance_data:
            d_ene = sum(d.calc(active_pos) for d in self.distance_data if d.is_valid())
            print(f"  dist   E={d_ene:.5f}")
