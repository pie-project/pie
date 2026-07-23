use plex::{Host, Policy, ScheduleContext, SchedulePlan, ScheduleSelection, State};

struct BadBudget;

impl Policy for BadBudget {
    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        Ok(SchedulePlan {
            selections: (!ctx.runnable.is_empty())
                .then(|| ScheduleSelection {
                    requests: vec![0],
                    token_budgets: vec![u32::MAX],
                })
                .into_iter()
                .collect(),
        })
    }
}

plex::export_policy!(BadBudget);
