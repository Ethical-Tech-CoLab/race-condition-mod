# Architecture Documentation Map

Detailed design notes for the simulation's core components. Use the map below to
find documentation for the area you care about.

## Map

### Agent network

How agents communicate and coordinate.

- [Agent Architecture](agent_architecture.md): topology of the ADK/A2A network.
- [Runner Decision Model](runner_decision_model.md): how an agent decides
  anything — the four layers, what the LLM actually controls, the seams used to
  generalise the engine across scenarios, and why per-agent inference does not
  scale.
- [Communication Protocol](communication_protocol.md): wire schema, handshake,
  and multi-session routing.
- [Route Planning](route_planning.md): GIS-based marathon route generation.

### Standards

- [A2UI Protocol](a2ui_protocol.md): agent-to-UI communication.
- [Port Mapping](ports.md): port assignments across local services.

## Suggested reading order for newcomers

1. [Project README](../../README.md) and the `getting-started` skill in
   `.claude/skills/`.
2. [Agent Architecture](agent_architecture.md).
3. [A2UI Protocol](a2ui_protocol.md).
4. [Implementing Skills](../guides/implementing_skills.md).
