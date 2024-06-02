from typing import Dict, Any

import pandas as pd

from .policy import AutoDaTransform, AutoDaSubPolicy, AutoDaPolicy

from .globals import (
    POLICY_PREFIX,
    PROB_PREFIX,
    LEVEL_PREFIX,
    POLICY_PREFIX_DF,
    PROB_PREFIX_DF,
    LEVEL_PREFIX_DF
)


def _decode_fastaa_policy(
    policy_config: Dict[str, Any],
    num_sub_policies: int,
    num_ops: int,
    policy_prefix: str,
    prob_prefix: str,
    level_prefix: str
) -> AutoDaPolicy:
    policy = AutoDaPolicy()
    for i in range(num_sub_policies):
        sub_policy = AutoDaSubPolicy()
        for j in range(num_ops):
            op_idx = policy_config[f'{policy_prefix}_{i}_{j}']
            op_prob = policy_config[f'{prob_prefix}_{i}_{j}']
            op_level = policy_config[f'{level_prefix}_{i}_{j}']

            sub_policy.append_transform(AutoDaTransform(op_idx, op_prob, op_level))

        policy.append_sub_policy(sub_policy)

    return policy


def decode_fastaa_policy_config(
    policy_config: Dict[str, Any],
    num_sub_policies: int,
    num_ops: int
) -> AutoDaPolicy:
    return _decode_fastaa_policy(
        policy_config=policy_config,
        num_sub_policies=num_sub_policies,
        num_ops=num_ops,
        policy_prefix=POLICY_PREFIX,
        prob_prefix=PROB_PREFIX,
        level_prefix=LEVEL_PREFIX
    )


def decode_fastaa_policy_df(policies_df: pd.DataFrame) -> AutoDaPolicy:
    subpol_filter = [col for col in policies_df.columns if col.startswith(POLICY_PREFIX_DF)]

    num_sub_policies = len(set(int(col.split('_')[1]) for col in subpol_filter))
    num_ops = len(set(int(col.split('_')[2]) for col in subpol_filter))

    policy = AutoDaPolicy()

    for index, row in policies_df.iterrows():
        row_policy = _decode_fastaa_policy(
            policy_config=row,
            num_sub_policies=num_sub_policies,
            num_ops=num_ops,
            policy_prefix=POLICY_PREFIX_DF,
            prob_prefix=PROB_PREFIX_DF,
            level_prefix=LEVEL_PREFIX_DF
        )

        for sub_policy in row_policy.sub_policies:
            policy.append_sub_policy(sub_policy)

    return policy
