

class Restriction(object):
    """The terminology used for these restriction types corresponds to a portfolio optimizer where
    the state vector consists of positions in financial instruments, but the concepts they represent
    are applicable to any sort of optimization.
    """
    NONE = 0
    HOLDLONG = 1    # hold from going longer
    HOLDSHORT = 2   # hold from going shorter
    HOLDBOTH = 3
    CLOSELONG = 4   # if long, liquidate
    CLOSESHORT = 5  # if short, liquidate
    CLOSEBOTH = 6
    PREVENTBUY = 7  # regardless of position, prevent buying
    PREVENTSELL = 8 # regardless of position, prevent selling
    UNTRADABLE = 9

    @staticmethod
    def constraints_satisfied(r, posmv, ordmv):
        """Test whether constraints are satisfied for a given proposed trade.
        @param r:  Restriction type.
        @param posmv:  Position (in currency units).
        @param ordmv:  Proposed order (in currency units).
        """
        if r == Restriction.NONE:
            return True
        if r == Restriction.UNTRADABLE:
            return False
        if r == Restriction.PREVENTBUY:
            return (ordmv <= 0)
        if r == Restriction.PREVENTSELL:
            return (ordmv >= 0)
        stepped = posmv + ordmv  # currency units
        if r == Restriction.HOLDLONG:
            return (stepped <= 0 or stepped <= posmv)
        if r == Restriction.HOLDSHORT:
            return (stepped >= 0 or stepped >= posmv)
        if r == Restriction.HOLDBOTH:
            return (stepped <= 0 or stepped <= posmv) and (stepped >= 0 or stepped >= posmv)
        return True
