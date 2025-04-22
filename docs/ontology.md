# Financial Knowledge Graph Ontology

This document defines the ontology used for the financial knowledge graph.

## Nodes (Entity Types)

- **Entity** (top-level class)
  - Base class for all entities in the knowledge graph
- **Company** (main entity, subclass of Entity)
  - The primary entity type representing business organizations
  - _Optional subclasses_: PublicCompany, PrivateCompany

## Edges (Relationship Types)

1. **hasDebtHolder**

   - Represents debt relationships between entities
   - Direction: Company → Entity (the entity holds debt from the company)

2. **hasEquityStakeIn**

   - Represents ownership or investment relationships
   - Direction: Entity → Company (the entity has equity in the company)

3. **mergedWith**

   - Represents merger transactions between companies
   - Direction: Company ↔ Company (bidirectional relationship)

4. **acquired**
   - Represents acquisition transactions
   - Direction: Company → Company (one company acquired the other)

## Attributes

### For Companies (Node Attributes)

- **companyName** (string)
  - The official name of the company
- **ticker** (string)
  - Stock symbol for public companies
  - Only applicable for publicly traded companies
- **industry** (string)
  - The business sector or industry the company operates in
- **country** (string)
  - Country where the company is headquartered or incorporated

### For Relationships (Edge Attributes)

- **valueAmount** (float)
  - The monetary value of the transaction
  - Used in acquisitions, mergers, debt, and equity relationships
- **percentage** (float)
  - The percentage of ownership or stake
  - Primarily used with hasEquityStakeIn relationships
- **transactionDate** (date)
  - The date when the transaction or relationship was established
  - Used across all relationship types
