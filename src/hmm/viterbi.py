"""HMM/Viterbi route reconstruction.

decode(candidates, probs, graph) -> dict[(trip_id, obs_idx)] -> arc_id
  hidden state   : candidate road arc at each observation
  emission       : classifier probability (log)
  transition     : road-network connectivity / distance consistency
TODO: implement, handle empty candidate sets and GPS gaps.
"""


def decode(candidates, probs, graph):
    raise NotImplementedError
