# AgentSociety Community

A community library for customizing agent/block for [AgentSociety](https://github.com/tsinghua-fib-lab/AgentSociety).

## How to add a new block

1. Add a new file in the directory to define your block class.
2. add a _import_xxx function to import the block class.
3. add a __getattr__ function to lazy import the block class.
4. add the block class to __all__ variable.
5. add the block class to the return value of get_type_to_cls_dict function.

## How to add a new agent

1. Add a new file in the directory to define your agent class.
2. add a _import_xxx function to import the agent class.
3. add a __getattr__ function to lazy import the agent class.
4. add the agent class to __all__ variable.
5. add the agent class to the return value of get_type_to_cls_dict function.
