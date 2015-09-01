import unittest
import numpy as np
import time
from opt import *


class OptimizerTests(unittest.TestCase):
    """nosetests --nologcapture tests/test_opt.py:OptimizerTests.test_optimize
    """

    def setUp(self):
        float_t = OptimizerBase._float_t
        days_per_year = 252

        self.optin = {}
        optin = self.optin

        # instruments
        ninsts = 2
        optin['resid_vars'] = np.array([0.22, 0.17], dtype=float_t)**2 / days_per_year
        optin['bod_positions'] = np.array([100, 0], dtype=float_t)  # shares units
        optin['hmaxs'] = np.array([1e6] * ninsts, dtype=float_t)  # currency units
        optin['hmins'] = -optin['hmaxs']
        optin['smaxs'] = np.array([25e3] * ninsts, dtype=float_t)
        optin['smins'] = -optin['smaxs']
        optin['mdvs'] = np.array([1e6, 1e5], dtype=float_t)  # shares units (because "volumes" are shares)
        optin['round_lots'] = np.array([100, 100], dtype=float_t)

        # risk model
        nfactors = 2
        optin['factor_vars'] = np.array([0.02, 0.09])**2 / days_per_year
        optin['factor_corrs'] = np.ones((nfactors, nfactors), dtype=float_t)
        optin['factor_corrs'][0, 1] = 0.2
        optin['factor_corrs'][1, 0] = 0.2

        optin['loadings'] = [list()] * ninsts
        optin['loadings'][0] = [(0, 1.1), (1, 0.5)]
        optin['loadings'][1] = [(0, 0.9), (1, 1.5)]

        # parameters
        hzon = 10.0
        optin['horizon'] = hzon
        optin['mvk'] = 2e-4  # using a large value for this test so that risk plays a part in this small portfolio
        optin['convergence_threshold'] = 0.0015  # = 15 bp ???  down to 4 ???  nope - bad idea
        optin['mdv_position_limit'] = 0.75
        optin['mdv_trading_limit'] = 0.05
        # optin['per_optimization_mdv_trading_limit'] = 0.0001  # s.t. the 1000'th name has a limit of $9k or ~200 shares, see constrained_opt_sizes.xlsx

        optin['maxgmv'] = 32.6e3  # 32.6 * 0.98 = 31.948 (so that the maxgmv penalty gets exercised)
        optin['maxgmv_buffer'] = 0.02
        optin['maxgmv_penaltyrate'] = 0.08

        # initialize optimizer
        self.optimizer = PortfolioOptimizer(**optin)

        # portfolio
        self.istate = []
        self.istate.append(PortfolioOptimizer.StateElem(optin['bod_positions'][0], 20.0,  0.009 / hzon, Restriction.NONE))
        self.istate.append(PortfolioOptimizer.StateElem(optin['bod_positions'][1], 50.0, -0.015 / hzon, Restriction.NONE))

    def tearDown(self):
        super().tearDown()
        delattr(self, 'optin')
        delattr(self, 'optimizer')
        delattr(self, 'istate')

    def test_optimize(self):
        """Test a basic optimization."""
        fstate = self.optimizer.optimize(self.istate)
        logging.debug(fstate)
        self.optimizer.log_optimization_stats()
        self.assertAlmostEqual(self.optimizer.futil.total, 222.84, places=2)

    def test_hmax(self):
        """Test hard max position limit."""
        self.optin['hmaxs'][0] = 1980  # inst[0] has a price of 20 so 1980 is 99 shares
        self.optin['smaxs'][0] = 1970  # use something less than the hmax to ensure lower smax doesn't prevent trade
        self.optimizer = PortfolioOptimizer(**self.optin)
        fstate = self.optimizer.optimize(self.istate)
        self.optimizer.log_optimization_stats()
        self.assertAlmostEqual(fstate[0].shares, 99)
        self.assertAlmostEqual(self.optimizer.futil.total, 167.05, places=2)

    def test_smax(self):
        """Test soft max position limit."""
        self.optin['smaxs'][0] = 1980
        self.optimizer = PortfolioOptimizer(**self.optin)
        fstate = self.optimizer.optimize(self.istate)
        self.optimizer.log_optimization_stats()
        self.assertAlmostEqual(fstate[0].shares, 100)
        self.assertAlmostEqual(self.optimizer.futil.total, 167.24, places=2)

    def restriction_subtest(self, i, r, u):
        self.istate[i].restriction = r
        self.optimizer.optimize(self.istate)
        self.optimizer.log_optimization_stats()
        self.assertAlmostEqual(self.optimizer.futil.total, u, places=2)

    def test_restrictions(self):
        """Test restriction types."""
        self.restriction_subtest(0, Restriction.HOLDBOTH  , 167.24)  # optimal inst 0 position is 600 shares
        self.restriction_subtest(0, Restriction.HOLDLONG  , 167.24)
        self.restriction_subtest(0, Restriction.UNTRADABLE, 167.24)
        self.restriction_subtest(0, Restriction.PREVENTBUY, 167.24)
        self.restriction_subtest(0, Restriction.HOLDSHORT , 222.84)
        self.restriction_subtest(0, Restriction.CLOSESHORT, 222.84)
        self.restriction_subtest(0, Restriction.CLOSELONG , 146.28)
        self.restriction_subtest(0, Restriction.CLOSEBOTH , 146.28)
        self.istate[0].restriction = Restriction.NONE
        self.restriction_subtest(1, Restriction.HOLDBOTH  ,  49.28)
        self.restriction_subtest(1, Restriction.HOLDSHORT ,  49.28)
        self.restriction_subtest(1, Restriction.UNTRADABLE,  49.28)
        self.restriction_subtest(1, Restriction.PREVENTSELL, 49.28)

    def test_maxgmv_penalty(self):
        """Test max GMV penalty."""
        self.optin['maxgmv'] = 33e3  # i.e. no effect b/c GMV gets up to 32e3 w/out any constraint and 32e3 < 33e3 * 0.98
        self.optimizer = PortfolioOptimizer(**self.optin)
        self.optimizer.optimize(self.istate)
        self.optimizer.log_optimization_stats()
        self.assertAlmostEqual(self.optimizer.futil.total, 223.18, places=2)
        self.assertAlmostEqual(self.optimizer.futil.maxgmv_penalty, 0.0)

    def test_fast_marginal_objective(self):
        """Test that the fast implementation of marginal_objective in PortfolioOptimizer is both faster and correct."""
        # http://stackoverflow.com/questions/1557571/how-to-get-time-of-a-python-program-execution
        # cProfile doesn't work w/ multi-threaded code b/c it only times what happens in Python
        tdur0 = 0
        tdur1 = 0
        reps = 10
        for i in range(reps):
            tstart = time.time()
            self.optimizer.optimize(self.istate)
            futil0 = self.optimizer.futil.total
            tdur0 += time.time() - tstart
            if i == 0: self.optimizer.log_optimization_stats()

            # delete fast marginal_objective implementation of of PortfolioOptimizer
            opt_t = type(self.optimizer)
            orig = opt_t.marginal_objective
            delattr(opt_t, 'marginal_objective')

            # time using slower base class implementation
            tstart = time.time()
            self.optimizer.optimize(self.istate)
            futil1 = self.optimizer.futil.total
            tdur1 += time.time() - tstart

            opt_t.marginal_objective = orig
            if i == 0: self.optimizer.log_optimization_stats()

        logging.debug('fast/slow (ms): {:.3f}/{:.3f} = {}%'.format(tdur0 * 1e3 / reps, tdur1 * 1e3 / reps, int(tdur0 / tdur1 * 100)))
        self.assertLess(tdur0, tdur1)
        self.assertEqual(futil0, futil1)

    def test_per_optimization_mdv_trading_limit(self):
        """Test per optimization trading limit."""
        self.optin['per_optimization_mdv_trading_limit'] = 0.00035
        self.optimizer = PortfolioOptimizer(**self.optin)
        fstate = self.optimizer.optimize(self.istate)
        self.assertAlmostEqual(fstate[0].shares, 400)
        self.assertAlmostEqual(fstate[1].shares, -100)
        self.assertAlmostEqual(self.optimizer.futil.total, 115.92, places=2)




        # logging.debug(self.optimizer.fstate)
        # self.optimizer.log_optimization_stats()
        # self.assertTrue(False)





if __name__ == '__main__':
    unittest.main()