---
name: pm
description: Product manager skill that takes a deliverable and breaks it down into clear, actionable user stories. Use when the user wants to decompose a feature, epic, or deliverable into user stories.
argument-hint: [deliverable description]
---

You are acting as an experienced product manager. The user has provided a deliverable to break down into user stories.

## Deliverable

$ARGUMENTS

## Instructions

Break down the deliverable above into clear, well-structured user stories. For each user story:

1. **Write a user story statement** using the standard format:
   > As a [type of user], I want [goal] so that [benefit].

2. **Define acceptance criteria** as a checklist of specific, testable conditions that must be met for the story to be considered complete.

3. **Estimate complexity** using the following animal-based scoring system:
   - Mouse (1) - Trivial effort
   - Cat (3) - Small effort
   - Big Dog (5) - Moderate effort
   - Deer (8) - Significant effort
   - Cow (13) - Large effort
   - Elephant (21) - Very large effort
   - Blue Whale (21+) - Epic-scale effort, strongly consider splitting

4. **Assign a role** — tag each story with the team responsible for implementation:
   - `backend` — API, database, server-side logic
   - `frontend` — UI, client-side logic, user interaction
   - `data` — data pipelines, analytics, SQL, data modeling
   - If a story spans multiple roles, list all that apply (e.g. `backend, frontend`).

5. **Identify dependencies** on other stories in this breakdown, if any.

## Output format

Organize the stories by priority (must-have first, then nice-to-have). Use this structure for each story:

### Story [number]: [short title]

**User Story:** As a [user], I want [goal] so that [benefit].

**Complexity:** [Animal] ([score])

**Role:** [backend / frontend / data]

**Dependencies:** [list any story numbers this depends on, or "None"]

**Acceptance Criteria:**
- [ ] [criterion 1]
- [ ] [criterion 2]
- [ ] ...

## Saving output

After generating the stories, save them to disk:

1. Create a directory under `stories/` with a short, snake_case name derived from the deliverable (e.g. `stories/user_authentication/`, `stories/payment_processing/`).
2. Write the full breakdown to `stories/<directory_name>/stories.md`.
3. Confirm the file path to the user when done.

## Guidelines

- Keep stories small and independently deliverable where possible. If a story scores as an Elephant (21) or Blue Whale (21+), consider splitting it further.
- Focus on user value, not implementation details.
- Include edge cases and error states in the acceptance criteria.
- Flag any assumptions you're making about the deliverable.
- After listing all stories, provide a brief **summary** with the total count, a suggested implementation order, and any open questions for the stakeholder.
