/**
 * Predefined Grammar Configurations
 *
 * Ready-to-use grammar configurations for common architectural patterns
 */

import { GrammarConfig, SimilarityAlgorithm } from './types'

/**
 * Clean Architecture grammar for code
 */
export const CLEAN_ARCHITECTURE_GRAMMAR: GrammarConfig = {
  roles: {
    Subject: {
      values: ["DbAddAccount", "RemoteAddAccount", "DbLoadSurvey", "RemoteLoadSurvey", "DbAuthentication", "RemoteAuthentication"],
      required: true,
      description: "The main actor/component performing the action"
    },
    Verb: {
      values: ["add", "delete", "update", "load", "save", "authenticate", "validate"],
      required: true,
      description: "The action being performed"
    },
    Object: {
      values: ["Account.Params", "Survey.Params", "User.Params", "Entity.Data", "Auth.Credentials"],
      required: false,
      description: "The data being acted upon"
    },
    Adverb: {
      values: ["Hasher", "Repository", "ApiAdapter", "Validator", "Encrypter", "TokenGenerator"],
      required: false,
      multiple: true,
      description: "Modifiers that describe how the action is performed"
    },
    Context: {
      values: ["Controller", "MainFactory", "Service", "UseCase", "Presenter"],
      required: false,
      description: "The architectural layer/context"
    }
  },
  structuralRules: [
    {
      name: "VerbObjectAlignment",
      validate: (s) => {
        if (s.Verb === "authenticate" && s.Object && !s.Object.includes("Auth")) {
          return false
        }
        return true
      },
      message: "Verb 'authenticate' requires an Auth-related Object"
    }
  ],
  options: {
    similarityThreshold: 0.65,
    similarityAlgorithm: SimilarityAlgorithm.HYBRID,
    enableCache: true,
    autoRepair: true,
    maxSuggestions: 3,
    caseSensitive: false
  }
}

/**
 * HTTP API grammar
 */
export const HTTP_API_GRAMMAR: GrammarConfig = {
  roles: {
    Method: {
      values: ["GET", "POST", "PUT", "PATCH", "DELETE"],
      required: true,
      description: "HTTP method"
    },
    Resource: {
      values: ["/users", "/posts", "/comments", "/auth", "/profiles"],
      required: true,
      description: "API resource path"
    },
    Status: {
      values: ["200", "201", "400", "401", "403", "404", "500"],
      required: false,
      description: "HTTP status code"
    },
    Handler: {
      values: ["Controller", "Middleware", "Guard", "Interceptor"],
      required: false,
      multiple: true,
      description: "Request handlers"
    }
  },
  options: {
    similarityThreshold: 0.7,
    similarityAlgorithm: SimilarityAlgorithm.LEVENSHTEIN,
    caseSensitive: true
  }
}
