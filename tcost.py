

class TCostCalculator(object):
    """Interface class for transactions cost calculation.
    @TODO: Use 80 bp for annual carry in lieu of actual borrow rates.
    @TODO: Use 0.0009 $/share for commission cost, but implement a per_trade commission parameter also.
    """
    def __call__(self, ipos, fpos, price, mdv, hzon):
        return (self._slippage(fpos - ipos, price, mdv) +
                self._commissions(fpos - ipos) +
                self._holding_costs(fpos, price, hzon) +
                self._liquidation_costs(fpos, price, mdv, hzon))

    def marginal(self, ipos, fpos, step, price, mdv, hzon):
        return (self(ipos, fpos + step, price, mdv, hzon) -
                self(ipos, fpos       , price, mdv, hzon))

    def marginal_slippage(self, ipos, fpos, step, price, mdv):
        return (self._slippage(fpos - ipos + step, price, mdv) -
                self._slippage(fpos - ipos       , price, mdv))

    def _slippage(self, step, price, mdv):
        return 0.0
    def _commissions(self, step):
        return 0.0
    def _holding_costs(self, pos, price, hzon):
        return 0.0
    def _liquidation_costs(self, pos, price, mdv, hzon):
        return self._slippage(pos, price, mdv * hzon)