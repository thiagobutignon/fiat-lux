/**
 * Internationalization (i18n) Support
 * Multi-language support for Cognitive OS
 * Initial languages: en (English), pt (Portuguese), es (Spanish)
 */

// ============================================================
// TYPES
// ============================================================

export type SupportedLocale = 'en' | 'pt' | 'es' | 'fr' | 'de' | 'ja' | 'zh';

export interface LocaleStrings {
  // Technique categories
  categories: {
    gaslighting: string;
    triangulation: string;
    love_bombing: string;
    darvo: string;
    word_salad: string;
    temporal_manipulation: string;
    boundary_violation: string;
    flying_monkeys: string;
    projection: string;
    silent_treatment: string;
    hoovering: string;
    smear_campaign: string;
    future_faking: string;
    moving_goalposts: string;
    emotional_blackmail: string;
  };

  // Dark Tetrad traits
  dark_tetrad: {
    narcissism: string;
    machiavellianism: string;
    psychopathy: string;
    sadism: string;
  };

  // Detection messages
  detection: {
    detected: string;
    confidence: string;
    linguistic_analysis: string;
    phonemes_matched: string;
    morphemes_matched: string;
    syntax_matched: string;
    semantics_matched: string;
    pragmatics_matched: string;
    dark_tetrad_profile: string;
    no_manipulation_detected: string;
    high_confidence_alert: string;
  };

  // Neurodivergent protection
  neurodivergent: {
    warning: string;
    markers_detected: string;
    autism_markers: string;
    adhd_markers: string;
    threshold_increased: string;
  };

  // Constitutional validation
  constitutional: {
    compliant: string;
    violations: string;
    warnings: string;
    privacy_protected: string;
    transparency_ensured: string;
    no_diagnosis: string;
  };

  // Common words
  common: {
    yes: string;
    no: string;
    unknown: string;
    loading: string;
    error: string;
    success: string;
    high: string;
    medium: string;
    low: string;
    critical: string;
  };
}

// ============================================================
// ENGLISH (en)
// ============================================================

const en: LocaleStrings = {
  categories: {
    gaslighting: 'Gaslighting',
    triangulation: 'Triangulation',
    love_bombing: 'Love Bombing',
    darvo: 'DARVO',
    word_salad: 'Word Salad',
    temporal_manipulation: 'Temporal Manipulation',
    boundary_violation: 'Boundary Violation',
    flying_monkeys: 'Flying Monkeys',
    projection: 'Projection',
    silent_treatment: 'Silent Treatment',
    hoovering: 'Hoovering',
    smear_campaign: 'Smear Campaign',
    future_faking: 'Future Faking',
    moving_goalposts: 'Moving Goalposts',
    emotional_blackmail: 'Emotional Blackmail'
  },

  dark_tetrad: {
    narcissism: 'Narcissism',
    machiavellianism: 'Machiavellianism',
    psychopathy: 'Psychopathy',
    sadism: 'Sadism'
  },

  detection: {
    detected: 'Detected',
    confidence: 'Confidence',
    linguistic_analysis: 'Linguistic Analysis (Chomsky Hierarchy)',
    phonemes_matched: 'Phonemes matched',
    morphemes_matched: 'Morphemes matched',
    syntax_matched: 'Syntax patterns matched',
    semantics_matched: 'Semantic meaning matched',
    pragmatics_matched: 'Intent/pragmatics matched',
    dark_tetrad_profile: 'Dark Tetrad Profile',
    no_manipulation_detected: 'No manipulation detected',
    high_confidence_alert: 'High-confidence manipulation detected'
  },

  neurodivergent: {
    warning: 'Neurodivergent communication patterns detected',
    markers_detected: 'Markers detected',
    autism_markers: 'Autism-related communication patterns',
    adhd_markers: 'ADHD-related communication patterns',
    threshold_increased: 'Confidence threshold increased for safety'
  },

  constitutional: {
    compliant: 'Constitutionally compliant',
    violations: 'Constitutional violations',
    warnings: 'Warnings',
    privacy_protected: 'Privacy protected - no personal data stored',
    transparency_ensured: 'All detections are explainable',
    no_diagnosis: 'Pattern detection only - not a diagnosis'
  },

  common: {
    yes: 'Yes',
    no: 'No',
    unknown: 'Unknown',
    loading: 'Loading',
    error: 'Error',
    success: 'Success',
    high: 'High',
    medium: 'Medium',
    low: 'Low',
    critical: 'Critical'
  }
};

// ============================================================
// PORTUGUESE (pt)
// ============================================================

const pt: LocaleStrings = {
  categories: {
    gaslighting: 'Gaslighting',
    triangulation: 'Triangulação',
    love_bombing: 'Bombardeio de Amor',
    darvo: 'DARVO',
    word_salad: 'Salada de Palavras',
    temporal_manipulation: 'Manipulação Temporal',
    boundary_violation: 'Violação de Limites',
    flying_monkeys: 'Macacos Voadores',
    projection: 'Projeção',
    silent_treatment: 'Tratamento Silencioso',
    hoovering: 'Aspiração',
    smear_campaign: 'Campanha de Difamação',
    future_faking: 'Futuro Falso',
    moving_goalposts: 'Mudança de Objetivo',
    emotional_blackmail: 'Chantagem Emocional'
  },

  dark_tetrad: {
    narcissism: 'Narcisismo',
    machiavellianism: 'Maquiavelismo',
    psychopathy: 'Psicopatia',
    sadism: 'Sadismo'
  },

  detection: {
    detected: 'Detectado',
    confidence: 'Confiança',
    linguistic_analysis: 'Análise Linguística (Hierarquia de Chomsky)',
    phonemes_matched: 'Fonemas correspondentes',
    morphemes_matched: 'Morfemas correspondentes',
    syntax_matched: 'Padrões sintáticos correspondentes',
    semantics_matched: 'Significado semântico correspondente',
    pragmatics_matched: 'Intenção/pragmática correspondente',
    dark_tetrad_profile: 'Perfil Dark Tetrad',
    no_manipulation_detected: 'Nenhuma manipulação detectada',
    high_confidence_alert: 'Manipulação de alta confiança detectada'
  },

  neurodivergent: {
    warning: 'Padrões de comunicação neurodivergente detectados',
    markers_detected: 'Marcadores detectados',
    autism_markers: 'Padrões de comunicação relacionados ao autismo',
    adhd_markers: 'Padrões de comunicação relacionados ao TDAH',
    threshold_increased: 'Limiar de confiança aumentado para segurança'
  },

  constitutional: {
    compliant: 'Constitucionalmente compatível',
    violations: 'Violações constitucionais',
    warnings: 'Avisos',
    privacy_protected: 'Privacidade protegida - nenhum dado pessoal armazenado',
    transparency_ensured: 'Todas as detecções são explicáveis',
    no_diagnosis: 'Apenas detecção de padrões - não é um diagnóstico'
  },

  common: {
    yes: 'Sim',
    no: 'Não',
    unknown: 'Desconhecido',
    loading: 'Carregando',
    error: 'Erro',
    success: 'Sucesso',
    high: 'Alto',
    medium: 'Médio',
    low: 'Baixo',
    critical: 'Crítico'
  }
};

// ============================================================
// SPANISH (es)
// ============================================================

const es: LocaleStrings = {
  categories: {
    gaslighting: 'Gaslighting',
    triangulation: 'Triangulación',
    love_bombing: 'Bombardeo de Amor',
    darvo: 'DARVO',
    word_salad: 'Ensalada de Palabras',
    temporal_manipulation: 'Manipulación Temporal',
    boundary_violation: 'Violación de Límites',
    flying_monkeys: 'Monos Voladores',
    projection: 'Proyección',
    silent_treatment: 'Tratamiento Silencioso',
    hoovering: 'Aspiración',
    smear_campaign: 'Campaña de Difamación',
    future_faking: 'Futuro Falso',
    moving_goalposts: 'Cambio de Objetivos',
    emotional_blackmail: 'Chantaje Emocional'
  },

  dark_tetrad: {
    narcissism: 'Narcisismo',
    machiavellianism: 'Maquiavelismo',
    psychopathy: 'Psicopatía',
    sadism: 'Sadismo'
  },

  detection: {
    detected: 'Detectado',
    confidence: 'Confianza',
    linguistic_analysis: 'Análisis Lingüístico (Jerarquía de Chomsky)',
    phonemes_matched: 'Fonemas coincidentes',
    morphemes_matched: 'Morfemas coincidentes',
    syntax_matched: 'Patrones sintácticos coincidentes',
    semantics_matched: 'Significado semántico coincidente',
    pragmatics_matched: 'Intención/pragmática coincidente',
    dark_tetrad_profile: 'Perfil Dark Tetrad',
    no_manipulation_detected: 'No se detectó manipulación',
    high_confidence_alert: 'Manipulación de alta confianza detectada'
  },

  neurodivergent: {
    warning: 'Patrones de comunicación neurodivergente detectados',
    markers_detected: 'Marcadores detectados',
    autism_markers: 'Patrones de comunicación relacionados con autismo',
    adhd_markers: 'Patrones de comunicación relacionados con TDAH',
    threshold_increased: 'Umbral de confianza aumentado por seguridad'
  },

  constitutional: {
    compliant: 'Constitucionalmente compatible',
    violations: 'Violaciones constitucionales',
    warnings: 'Advertencias',
    privacy_protected: 'Privacidad protegida - no se almacenan datos personales',
    transparency_ensured: 'Todas las detecciones son explicables',
    no_diagnosis: 'Solo detección de patrones - no es un diagnóstico'
  },

  common: {
    yes: 'Sí',
    no: 'No',
    unknown: 'Desconocido',
    loading: 'Cargando',
    error: 'Error',
    success: 'Éxito',
    high: 'Alto',
    medium: 'Medio',
    low: 'Bajo',
    critical: 'Crítico'
  }
};

// ============================================================
// LOCALE REGISTRY
// ============================================================

const locales: Record<SupportedLocale, LocaleStrings> = {
  en,
  pt,
  es,
  fr: en, // TODO: Add French translations
  de: en, // TODO: Add German translations
  ja: en, // TODO: Add Japanese translations
  zh: en  // TODO: Add Chinese translations
};

// ============================================================
// I18N API
// ============================================================

let currentLocale: SupportedLocale = 'en';

/**
 * Set the current locale
 */
export function setLocale(locale: SupportedLocale): void {
  if (!locales[locale]) {
    console.warn(`Locale ${locale} not supported. Falling back to 'en'.`);
    currentLocale = 'en';
  } else {
    currentLocale = locale;
  }
}

/**
 * Get the current locale
 */
export function getLocale(): SupportedLocale {
  return currentLocale;
}

/**
 * Get locale strings for current locale
 */
export function t(): LocaleStrings {
  return locales[currentLocale];
}

/**
 * Get locale strings for a specific locale
 */
export function getStrings(locale: SupportedLocale): LocaleStrings {
  return locales[locale] || locales.en;
}

/**
 * Format a detection message with current locale
 */
export function formatDetectionMessage(
  techniqueName: string,
  confidence: number,
  locale?: SupportedLocale
): string {
  const strings = locale ? getStrings(locale) : t();
  return `${strings.detection.detected}: ${techniqueName} (${strings.detection.confidence}: ${(confidence * 100).toFixed(1)}%)`;
}

/**
 * Get all supported locales
 */
export function getSupportedLocales(): SupportedLocale[] {
  return Object.keys(locales) as SupportedLocale[];
}

/**
 * Check if a locale is supported
 */
export function isLocaleSupported(locale: string): locale is SupportedLocale {
  return locale in locales;
}
