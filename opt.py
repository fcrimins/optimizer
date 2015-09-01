"""opt.py
A generalized optimizer that has specific overriding for portfolio optimization (and which should eventually be
ported to Cython).
"""
import re
import copy
import functools
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
from restriction import Restriction
from tcost import TCostCalculator


class OptimizerBase(object):
    """Optimizer base class.  Declares some virtual functions and provides simplistic implementations
    for others.
    """
    _float_t = np.float64

    def __init__(self, **kws):
        """Copies keys out of kws and sets the corresponding attributes on self (with leading underscores).
        Also initializes an _optalg attribute to None if no 'optalg' attribute in kws.
        """
        # copy kws onto self
        for k, v in kws.items():
            setattr(self, '_' + k, v)
        if not hasattr(self, '_optalg'):
            self._optalg = None
            
    def choose_step(self, state):
        """Some optimization methods may only want to choose a single step at a time and then
        come back later once that step has been evaluated in order to iterate.
        """
        return self._optalg.choose_step(self, state)

    def _constraints_satisfied(self, state, step=None):
        """Leave step=None to check constraints for all elements of state.  Otherwise only
        check constraints corresponding to a change by step.
        """
        if step is None or isinstance(step, list):
            # for t in state: if not self._constraints_satisfied(self, step=t0): return False
            # the above would be a possibly suitable base class implementation of this function
            return NotImplementedError()

        # handle "restriction type" constraints, i.e. constraints that limit trading, which is why
        # "close type" Restrictions are not handled here because they potentially induce trading
        i, sz = step
        t = state[i]
        return Restriction.constraints_satisfied(t.restriction, t.shares * t.price, sz)

    def _satisfy_breached_constraints(self, state):
        """If any constraints are breached generate trades to satisfy them.  Also, mark such instruments
        as untradable so that attempted "breached constraints trades" aren't unwound.
        """
        for i, t in enumerate(state):
            if t.restriction != Restriction.UNTRADABLE:
                if (t.restriction == Restriction.CLOSELONG  and t > 0 or  # > not >= is important here
                    t.restriction == Restriction.CLOSESHORT and t < 0 or  # o/w we could get stuck at 0
                    t.restriction == Restriction.CLOSEBOTH):
                    sz = -t.shares * t.price

                    # don't check self._constraints_satisfied(state, step=(i, sz)) here as the "hard" constraints
                    # we're checking now should take precedence over the "soft" constraints that are checked
                    # inside _constraints_satisfied, e.g. if we've breached a hard max position limit then the
                    # required trade to get us to that limit will typically breach the lower soft position limit
                    if sz:
                        t += sz
                        t.restriction = Restriction.UNTRADABLE  # prevent unwinding
        return state

    def optimize(self, istate):
        """Run the optimization algorithm given istate and any other state attached to the
        Optimizer.
        @param istate: Initial state vector, the types of the elements of which will depend
            on the particular optimizer.
        @return: Final state.
        """
        self.istate = istate
        state = copy.deepcopy(istate)

        state = self._satisfy_breached_constraints(state)

        # iterate until choose_step returns None
        while True:
            # step can be an (index, step) pair or a list of steps corresponding to each element of state
            step = self.choose_step(state)
            if step is None:
                break
            if not self._constraints_satisfied(state, step=step):  # likely a redundant check
                raise RuntimeError('Chosen step {} does not satisfy constraints'.format(step))
            state = self._optalg.step_state(self, state, step)

        self.fstate = state
        return state

    def objective_func(self, state):
        raise NotImplementedError()

    def marginal_objective(self, state, step):
        """Note that better, more efficient implementations of this method are likely possible given
        a suitable objective function.
        @param state:
        @param step: Vector of currency unit steps (because OptimizerBase shouldn't know
            anything about shares/currencies).
        @return: Returns a floating point value representing the marginal utility.
        """
        if isinstance(step, list):
            stepped_state = [x + d for x, d in zip(state, step)]
            stepped_util = self.objective_func(stepped_state)
        else:
            assert(isinstance(step, tuple))
            state[step[0]] += step[1]  # incr and then decr to avoid having to deepcopy state (@TODO: not threadsafe)
            stepped_util = self.objective_func(state)
            state[step[0]] -= step[1]
        return stepped_util - self.objective_func(state)


class PortfolioOptimizer(OptimizerBase):
    """An optimizer for optimizing a portfolio.  The state vectors for this optimizer are lists
    of PortfolioOptimizer.StateElems that contain a position (in shares) along with a current
    (i.e. potentially intraday) price and forecast.
    """
    _TOL = 1e-4

    def __init__(self, **kws):
        super().__init__(**kws)
        if hasattr(self, '_optalg') and self._optalg is not None:
            raise ValueError('_optalg cannot be set in PortfolioOptimizer as it provides its own')
        self._optalg = GreedyOptAlg(self._convergence_threshold)
        if not hasattr(self, '_tcost_calculator'):
            self._tcost_calculator = TCostCalculator()
        self._sanity_checks()

        # preprocess the factor covariance matrix to limit the number of sqrts and products in marginal_objective_func
        factor_sigs = np.sqrt(self._factor_vars)
        # must transpose the first product so that numpy broadcasting will work along the correct dimensions
        self._factor_covars = (factor_sigs * self._factor_corrs).T * factor_sigs
        assert(np.allclose(self._factor_covars, self._factor_covars.T))  # assert(symmetric)

    def _sanity_checks(self):
        self._sanity_check_params()
        self._sanity_check_insts()
        self._sanity_check_factors()

    def _sanity_check_params(self):
        assert(self._convergence_threshold > 0.0)
        assert(self._convergence_threshold < 1.0)
        assert(self._mvk > 0.0)
        assert(self._horizon > 0)
        assert(self._horizon < 366)  # use calendar days just in case we ever decide to switch
        assert(self._mdv_position_limit < 2.0)
        assert(self._mdv_position_limit > 0.0)
        assert(self._mdv_trading_limit < self._mdv_position_limit)
        assert(self._mdv_trading_limit > 0.0)

        if hasattr(self, '_per_optimization_mdv_trading_limit'):
            assert(self._per_optimization_mdv_trading_limit < self._mdv_trading_limit)
            assert(self._per_optimization_mdv_trading_limit > 0.0)

        if hasattr(self, '_maxgmv') and np.isfinite(self._maxgmv):
            assert(self._maxgmv > np.max(self._smaxs))
            assert(self._maxgmv_buffer > 0.0)
            assert(self._maxgmv_buffer < 1.0)
            assert(self._maxgmv_penaltyrate > 0.0)
            assert(self._maxgmv_penaltyrate < 1.0)
        else:
            self._maxgmv = np.nan

    def _sanity_check_insts(self):
        ninsts = len(self._resid_vars)
        assert(ninsts == len(self._bod_positions))
        assert(ninsts == len(self._hmaxs))
        assert(ninsts == len(self._hmins))
        assert(ninsts == len(self._smaxs))
        assert(ninsts == len(self._smins))
        assert(ninsts == len(self._mdvs))
        assert(ninsts == len(self._round_lots))
        assert(ninsts == len(self._loadings))
        assert(all(x > 0.0 for x in self._mdvs if np.isfinite(x)))
        assert(all(h >= x for h, x in zip(self._hmaxs, self._smaxs)))
        assert(all(x >= s for x, s in zip(self._smaxs, self._smins)))
        assert(all(s >= n for s, n in zip(self._smins, self._hmins)))

    def _sanity_check_factors(self):
        nfactors = len(self._factor_vars)
        assert(nfactors == self._factor_corrs.shape[0])
        assert(np.allclose(self._factor_corrs, self._factor_corrs.T))  # symmetric correlation matrix

    def stepsize(self, state, i, direction):
        shrs = self._round_lots[i]  # @TODO: obviously computation of number of shares to step should be more complex
        return direction * state[i].price * shrs

    def _constraints_satisfied(self, state, step):
        """Test whether step satisfies the various hard/soft/restriction/etc. constraints.  For
        this implementation of this method, we assume the constraints are satisfied for the given
        state, so we only test the constraints having to do with the state elements corresponding
        to the step.
        @param state: State vector.
        @param step: (index, desired trade in currency units) tuple.
        @return: Boolean depending on whether step satisfies constraints.
        """
        if not super()._constraints_satisfied(state, step=step):
            return False
        i, sz = step  # sz is in currency units (i.e. market value)
        px = state[i].price
        newpos_mv = state[i].shares * px + sz
        trdtoday = newpos_mv - self._bod_positions[i] * px
        mdv_mv = self._mdvs[i] * px

        if hasattr(self, '_per_optimization_mdv_trading_limit'):
            trdopt_shs = newpos_mv / px - self.istate[i].shares  # traded this optimization
            if (np.fabs(trdopt_shs * px) > self._per_optimization_mdv_trading_limit * mdv_mv and
                np.fabs(trdopt_shs) > self._round_lots[i]):  # i.e. allow at least a single round lot
                return False

        return (newpos_mv <= self._smaxs[i] and
                newpos_mv >= self._smins[i] and  # @TODO: implement "constrained" single-optimization limits
                np.fabs(newpos_mv) <= self._mdv_position_limit * mdv_mv and
                np.fabs(trdtoday)  <= self._mdv_trading_limit  * mdv_mv)

    def _satisfy_breached_constraints(self, state):
        """If any constraints are breached generate trades to satisfy them.  Also, mark such instruments
        as untradable so that attempted "breached constraints trades" aren't unwound.
        """
        state = super()._satisfy_breached_constraints(state)
        for i, t in enumerate(state):
            if t.restriction != Restriction.UNTRADABLE:
                sz = 0.0
                if t > self._hmaxs[i]:
                    sz = self._hmaxs[i] - t.shares * t.price
                if t < self._hmins[i]:
                    sz = self._hmins[i] - t.shares * t.price

                # see note in base class implementation of this method about checking _constraints_satisfied here
                if sz:
                    t += sz
                    t.restriction = Restriction.UNTRADABLE  # prevent optimization from unwinding this trade
        return state

    def optimize(self, istate):
        """Overrides base class implementation so that trades that violate constraints can be
        forced prior to iterating.
        """
        self.objective_func(istate)  # record initial utility
        self.iutil = self.util

        fstate = super().optimize(istate)

        self.objective_func(fstate)  # record final utility
        self.futil = self.util

        return fstate

    def objective_func(self, state):
        """Compute typical mean-variance utility although marginal_objective_func will typically
        be used in place of this function in the PortfolioOptimizer.
        """
        hzon = self._horizon
        mu = 0.0
        rsig2 = 0.0
        tcost = 0.0
        for i, t in enumerate(state):
            pos_mv = t.shares * t.price
            mu    += t.forecast          * pos_mv
            rsig2 += self._resid_vars[i] * pos_mv * pos_mv

            # @TODO: in production we should only charge marginal slippage intraday and assume that intraday
            # prices reflect any other slippage that's occurred since the beginning of the day, we could
            # even make this function reflect that by attributing any price change since the beginning of the
            # day to slippage (that might make utility harder to understand though)
            # @TODO: additionally, at least while small, perhaps intraday prices (in simulation) should be
            # assumed to already contain adjustments for slippage due to others like us who are already
            # out there trading in the market -- nope, simulations will trade too much
            tcost += self._tcost_calculator(self._bod_positions[i], state[i].shares, state[i].price, self._mdvs[i], hzon)

        fexps = self._factor_exposures(state)
        fsig2 = 0.0  # np.inner(np.inner((fexps * np.sqrt(self._factor_vars)), self._factor_corrs), (fexps * np.sqrt(self._factor_vars)))
        for i, (expi, sig2i) in enumerate(zip(fexps, self._factor_vars)):
            if np.fabs(expi) > self._TOL:  # @TODO: consider enlarged 0 factor exposures [http://dev-app:8080/browse/CRIM-20]
                fsig2 += sig2i * expi * expi  # diagonal variance
                for j, sig2j in enumerate(self._factor_vars[:i]):  # note the slicing
                    fsig2 += 2 * self._factor_corrs[i, j] * np.sqrt(sig2i * sig2j) * expi * fexps[j]

        maxgmv_penalty = self._maxgmv_penalty(state)

        self.util = self.UtilityComponents()  # @TODO: not threadsafe
        self.util.forecast = mu * hzon
        self.util.resid_risk  = -self._mvk * rsig2 * hzon
        self.util.factor_risk = -self._mvk * fsig2 * hzon
        self.util.tcost = -tcost
        self.util.maxgmv_penalty = -maxgmv_penalty
        return self.util.total

    def marginal_objective(self, state, step):
        """Overrides slower (more general) base class implementation.
        Y = state (i.e. portfolio)
        X = step (i.e. proposed trade)
        MARGINAL_VAR(X | Y) = VAR(X + Y) - VAR(Y) = VAR(X) + 2 * COV(X, Y)

        Prior to switching from using fcorrs and sqrt(fvars) to using fcovars, this function was about
        33% faster than the base class implementation (for the simple test case in OptimizerTests).
        After switching, it is about 40% faster.
        """
        hzon = self._horizon
        i, sz = step
        t = state[i]
        mu    = t.forecast          * sz
        rsig2 = self._resid_vars[i] * sz * (sz + 2 * t.shares * t.price)

        # @TODO: see 2 @TODOs in objective_func corresponding to this line
        tcost = self._tcost_calculator.marginal(self._bod_positions[i], t.shares, sz / t.price, t.price, self._mdvs[i], hzon)

        fexps = self._factor_exposures(state)
        marg_fexps = np.zeros(len(fexps), dtype=self._float_t)
        for j, l in self._loadings[i]:
            marg_fexps[j] = l * sz

        fcov = lambda a, b: np.inner(np.inner(a, self._factor_covars), b)
        fsig2 = fcov(marg_fexps, marg_fexps) + 2 * fcov(marg_fexps, fexps)

        # @TODO: this could be smarter (also, changing state temporarily is not threadsafe)
        t += sz
        maxgmv_penalty = self._maxgmv_penalty(state)
        t -= sz
        maxgmv_penalty -= self._maxgmv_penalty(state)

        return (mu - self._mvk * (rsig2 + fsig2)) * hzon - tcost - maxgmv_penalty

    def _maxgmv_penalty(self, state):
        """Compute the max GMV penalty based on the _maxgmv, _maxgmv_penaltyrate, and _maxgmv_buffer
        attributes.
        """
        if not np.isfinite(self._maxgmv):
            return 0
        gmv = self._lmv(state) + self._smv(state)
        buffer_excess = gmv - self._maxgmv * (1 - self._maxgmv_buffer)
        if buffer_excess < 0:
            return 0
        if gmv >= self._maxgmv:
            return buffer_excess * self._maxgmv_penaltyrate
        penrate = (buffer_excess / gmv / self._maxgmv_buffer) * self._maxgmv_penaltyrate  # linearly interpolate rate
        return penrate * buffer_excess

    def _lmv(self, state):
        return sum(t.shares * t.price for t in state if t.shares > 0)

    def _smv(self, state):
        return -sum(t.shares * t.price for t in state if t.shares < 0)

    def _factor_exposures(self, state):
        """Compute and return factor exposures given state vector."""
        fexps = np.zeros(len(self._factor_vars), dtype=self._float_t)
        for t, ls in zip(state, self._loadings):
            pos_mv = t.shares * t.price
            if np.fabs(pos_mv) > 0:  # regardless of tradability
                for i, l in ls:
                    fexps[i] += l * pos_mv
        return fexps

    def log_optimization_stats(self):
        ntrades = sum(1 for i, f in zip(self.istate, self.fstate) if i.shares != f.shares)
        lmv = self._lmv(self.fstate)
        smv = self._smv(self.fstate)
        units = 1 # 1e3
        logging.info('Optimization: ntrades={} LMV($M)={:.3f} SMV($M)={:.3f} utility($K): i={:.2f} f={:.2f} fcast={:.2f} rrisk={:.2f} frisk={:.2f} tcost={:.2f}, penalty={:.2f}'.format(ntrades, lmv / 1e3 / units, smv / 1e3 / units, self.iutil.total / units, self.futil.total / units, self.futil.forecast / units, self.futil.resid_risk / units, self.futil.factor_risk / units, self.futil.tcost / units, self.futil.maxgmv_penalty / units))

    class UtilityComponents(object):
        __slots__ = ['forecast', 'resid_risk', 'factor_risk', 'tcost', 'maxgmv_penalty']
        @property
        def total(self):
            return sum(getattr(self, c) for c in self.__slots__)

    @functools.total_ordering
    class StateElem(object):
        """Objects of this type will be the elements of the state arrays for the PortfolioOptimizer."""
        __slots__ = ['shares', 'price', 'forecast', 'restriction']
        def __init__(self, *args):
            for i, v in enumerate(args):
                setattr(self, self.__slots__[i], v)

        def __iadd__(self, sz):
            """Update shares by sz / price.
            @param sz: In currency units and converted back to share units before updating self.shares.
            """
            self.shares += sz / self.price
            return self

        def __isub__(self, sz):
            return self.__iadd__(-sz)

        def __eq__(self, sz):
            return self.shares * self.price == sz

        def __lt__(self, sz):
            return self.shares * self.price < sz

        def __repr__(self):
            clsname = self.__class__.__name__
            suprepr = super().__repr__()
            m = re.search(r'^.+{}\sobject\s(.+)$'.format(clsname), suprepr)
            return '<{} ({}, {}, {}, {}) {}'.format(clsname, self.shares, self.price, self.forecast, self.restriction, m.group(1))


class GenericOptimizer(OptimizerBase):
    def __init__(self, **kws):
        super().__init__(**kws)
        if not hasattr(self, '_optalg'):
            raise ValueError('_optalg missing for GenericOptimizer')


class OptAlg(object):

    def choose_step(self, optimizer, state):
        raise NotImplmentedError()

    def step_state(self, optimizer, state, step):
        raise NotImplmentedError()
    

class GreedyOptAlg(OptAlg):
    """Greedy optimization."""
    def __init__(self, convergence_threshold):
        self._convergence_threshold = convergence_threshold

    def choose_step(self, optimizer, state):
        """Choose best step.  E.g. choose best 100-share order."""
        # @TODO: this loop can be done in parallel, use prange, could even select multiple/all bests to
        # expand in parallel against the start-of-optimization portfolio, orders that don't have positive
        # utility against the start-of-(constrained)-optimization portfolio shouldn't be issued anyway
        bestmutil = self._convergence_threshold  # best mutil_rate must be larger than convergence threshold
        beststep = (-1, 0)  # (index, step size in currency units) pair
        for i in range(len(state)):
            for direction in (-1, 1):
                sz = optimizer.stepsize(state, i, direction)  # step size in currency units
                if optimizer._constraints_satisfied(state, step=(i, sz)):
                    mutil = optimizer.marginal_objective(state, (i, sz))
                    mutil_rate = mutil / np.fabs(sz)
                    if (mutil_rate > bestmutil):
                        bestmutil = mutil_rate
                        beststep = (i, sz)
        return None if beststep[0] == -1 else beststep

    def step_state(self, optimizer, state, step):
        """Expand best direction until utility of marginal steps drops below convergence threshold."""
        if not optimizer._constraints_satisfied(state, step=step):
            return state
        state = list(state)  # @TODO: need deepcopy for this to be threadsafe (should we expand in parallel?)
        i = step[0]
        state[i] += step[1]  # implies that state elems are in currency units (or understand += at least)
        direction = (1 if step[1] > 0 else -1)
        while True:
            sz = optimizer.stepsize(state, i, direction)
            if not optimizer._constraints_satisfied(state, step=(i, sz)):
                return state
            # @TODO: could compute a (quadratic) gradient of mutil w.r.t. sz and jump to (half of) where
            # convergence_threshold will be reached as an attempt to speed up full-day optimizations
            mutil = optimizer.marginal_objective(state, (i, sz))
            if mutil / np.fabs(sz) < self._convergence_threshold:
                return state
            state[i] += sz


class SGDOptAlg(OptAlg):
    """Stochastic gradient descent."""
    pass


class PSOOptAlg(OptAlg):
    """Particle swarm optimization."""
    pass



if __name__ == "__main__":
    logging.error('See OptimizerTests')
