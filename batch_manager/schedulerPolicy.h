
#pragma once

#include "../executor/types.h"

namespace nexly::batch_manager::batch_scheduler
{

enum class SchedulerPolicy
{
    MAX_UTILIZATION,
    GUARANTEED_NO_EVICT,
};

SchedulerPolicy execToBatchManagerSchedPolicy(executor::SchedulerPolicy policy);

executor::SchedulerPolicy batchManagerToExecSchedPolicy(SchedulerPolicy policy);

}
