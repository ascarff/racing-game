This is a repo containing work done during the 17th Feb 2026 Steel City AI session on Agent Skills ([event info](https://alex-kelly.blog/Sheffield_event_info/event_4_agent_skills.html)).

I was all created using Claude Code, testing the ability of it and agents to run (mostly) solo.

It contains the skills files in `.claude/`, with:
- A PM skill, to create user stories (found in `stories/`)
- Backend, frontend and data science skills, to be able to build the project
- A build skill, to make an implementation plan and launch subagents with the dev skills to actually build the project.

I asked the PM to plan a simple 2d racing game, with a computer player powered with machine learning. It created the user stories found in `stories/2d_racing_game/stories.md`. I then asked the build agent to use subagents to build the project.

The resulting code is in `racing-game/` and `racing-game-ml/`.

The game is currently buggy as hell, but it does run! I'll come back and fix it one day...