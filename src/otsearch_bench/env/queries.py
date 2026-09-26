"""GraphQL documents for the Open Targets Platform API.

Every field name here was verified by live introspection of
https://api.platform.opentargets.org/api/v4/graphql (API 26.6.3, data release 26.06).
Each document is a named operation; the name is recorded in the cache's ``operation`` column.
"""

META = """
query Meta {
  meta {
    name
    apiVersion { x y z suffix }
    dataVersion { year month iteration }
  }
}
"""

MAP_IDS = """
query MapIds($terms: [String!]!, $entityNames: [String!]) {
  mapIds(queryTerms: $terms, entityNames: $entityNames) {
    mappings { term hits { id entity name score description } }
  }
}
"""

SEARCH = """
query Search($q: String!, $entityNames: [String!], $page: Pagination) {
  search(queryString: $q, entityNames: $entityNames, page: $page) {
    total
    hits { id entity name score description }
  }
}
"""

TARGET = """
query Target($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    symbolSynonyms { label source }
    nameSynonyms { label source }
  }
}
"""

DISEASE = """
query Disease($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    description
    dbXRefs
    synonyms { relation terms }
    therapeuticAreas { id name }
    parents { id name }
    ancestors
    descendants
  }
}
"""

DRUG = """
query Drug($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    drugType
    maximumClinicalStage
    description
    synonyms { label source }
    tradeNames { label source }
    mechanismsOfAction {
      rows { mechanismOfAction actionType targetName targets { id approvedSymbol } }
    }
    indications {
      count
      rows { id maxClinicalStage disease { id name } }
    }
  }
}
"""

# "Known drugs" is exposed as Target.drugAndClinicalCandidates in the current schema
# (the older Target.knownDrugs field no longer exists).
KNOWN_DRUGS = """
query KnownDrugs($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug { id name drugType }
        diseases { diseaseFromSource disease { id name } }
      }
    }
  }
}
"""

ASSOCIATED_DISEASES = """
query AssociatedDiseases(
  $ensemblId: String!
  $page: Pagination!
  $enableIndirect: Boolean
  $Bs: [String!]
) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: $page, enableIndirect: $enableIndirect, Bs: $Bs) {
      count
      rows {
        score
        disease { id name }
        datatypeScores { id score }
        datasourceScores { id score }
      }
    }
  }
}
"""

ASSOCIATED_TARGETS = """
query AssociatedTargets(
  $efoId: String!
  $page: Pagination!
  $enableIndirect: Boolean
  $Bs: [String!]
) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: $page, enableIndirect: $enableIndirect, Bs: $Bs) {
      count
      rows {
        score
        target { id approvedSymbol }
        datatypeScores { id score }
        datasourceScores { id score }
      }
    }
  }
}
"""

EVIDENCE = """
query Evidence(
  $ensemblId: String!
  $efoIds: [String!]!
  $datasourceIds: [String!]
  $size: Int
  $cursor: String
) {
  target(ensemblId: $ensemblId) {
    id
    evidences(efoIds: $efoIds, datasourceIds: $datasourceIds, size: $size, cursor: $cursor) {
      count
      cursor
      rows {
        id
        datasourceId
        datatypeId
        score
        resourceScore
        target { id approvedSymbol }
        disease { id name }
        drug { id name }
        clinicalStage
        literature
        studyId
        variantRsId
        confidence
        publicationYear
        releaseVersion
        directionOnTrait
        directionOnTarget
        trialStopReasonCategories
      }
    }
  }
}
"""
