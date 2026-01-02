//! Fast CSS selector parsing and matching for SVG style resolution
//!
//! Optimizations:
//! - Pre-parsed selectors stored in efficient structs
//! - Batch matching: check all rules for an element in one call
//! - Early rejection based on tag/id (most common filters)
//! - Indexed lookups for simple selectors (id, class, tag)

use pyo3::prelude::*;
use std::collections::HashMap;

/// A pre-parsed simple CSS selector (no combinators)
#[derive(Clone, Debug)]
struct ParsedSelector {
    tag: Option<String>,
    id: Option<String>,
    classes: Vec<String>,
    attrs: Vec<(String, Option<String>)>,
}

/// A full CSS rule with pre-parsed selector parts
#[derive(Clone, Debug)]
struct CssRule {
    /// Selector parts (for "div .foo #bar", this is [div_sel, foo_sel, bar_sel])
    parts: Vec<ParsedSelector>,
    /// Rule index for ordering
    index: usize,
    /// Specificity tuple (ids, classes+attrs, elements)
    specificity: (usize, usize, usize),
}

/// CSS matching engine with indexed lookups
#[pyclass]
pub struct CssMatcher {
    /// All CSS rules
    rules: Vec<CssRule>,
    /// Index: tag -> rule indices that require this tag
    tag_index: HashMap<String, Vec<usize>>,
    /// Index: id -> rule indices that require this id
    id_index: HashMap<String, Vec<usize>>,
    /// Index: class -> rule indices that require this class
    class_index: HashMap<String, Vec<usize>>,
    /// Rules that match any element (no tag/id/class requirement)
    universal_rules: Vec<usize>,
}

/// Parse a simple selector string into ParsedSelector
fn parse_simple_selector(selector: &str) -> ParsedSelector {
    if selector == "*" {
        return ParsedSelector {
            tag: None,
            id: None,
            classes: Vec::new(),
            attrs: Vec::new(),
        };
    }

    let mut remaining = selector.to_string();
    let mut required_id: Option<String> = None;
    let mut required_classes: Vec<String> = Vec::new();
    let mut required_attrs: Vec<(String, Option<String>)> = Vec::new();

    // Extract ID - find # not inside []
    let mut bracket_depth = 0;
    let mut id_start: Option<usize> = None;
    for (i, c) in remaining.chars().enumerate() {
        match c {
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            '#' if bracket_depth == 0 => {
                id_start = Some(i);
                break;
            }
            _ => {}
        }
    }

    if let Some(start) = id_start {
        // Find end of ID
        let after_hash = &remaining[start + 1..];
        let id_end = after_hash
            .find(|c: char| c == '.' || c == '#' || c == '[' || c == ':')
            .unwrap_or(after_hash.len());
        required_id = Some(after_hash[..id_end].to_string());
        remaining = format!("{}{}", &remaining[..start], &remaining[start + 1 + id_end..]);
    }

    // Extract classes
    loop {
        let mut bracket_depth = 0;
        let mut class_start: Option<usize> = None;
        for (i, c) in remaining.chars().enumerate() {
            match c {
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                '.' if bracket_depth == 0 => {
                    class_start = Some(i);
                    break;
                }
                _ => {}
            }
        }

        match class_start {
            Some(start) => {
                let after_dot = &remaining[start + 1..];
                let class_end = after_dot
                    .find(|c: char| c == '.' || c == '#' || c == '[' || c == ':')
                    .unwrap_or(after_dot.len());
                required_classes.push(after_dot[..class_end].to_string());
                remaining = format!("{}{}", &remaining[..start], &remaining[start + 1 + class_end..]);
            }
            None => break,
        }
    }

    // Extract attribute selectors [...]
    while let Some(start) = remaining.find('[') {
        if let Some(end) = remaining[start..].find(']') {
            let attr_content = &remaining[start + 1..start + end];
            if let Some(eq_pos) = attr_content.find('=') {
                let attr_name = attr_content[..eq_pos].trim_end_matches(|c| c == '~' || c == '|' || c == '^' || c == '$' || c == '*');
                let attr_val = attr_content[eq_pos + 1..].trim_matches(|c| c == '"' || c == '\'');
                required_attrs.push((attr_name.to_string(), Some(attr_val.to_string())));
            } else {
                required_attrs.push((attr_content.to_string(), None));
            }
            remaining = format!("{}{}", &remaining[..start], &remaining[start + end + 1..]);
        } else {
            break;
        }
    }

    // Remaining is tag name
    let tag = remaining.trim();
    let required_tag = if tag.is_empty() || tag == "*" {
        None
    } else {
        Some(tag.to_string())
    };

    ParsedSelector {
        tag: required_tag,
        id: required_id,
        classes: required_classes,
        attrs: required_attrs,
    }
}

/// Calculate CSS specificity
fn calculate_specificity(selector: &str) -> (usize, usize, usize) {
    let mut ids = 0;
    let mut classes_attrs = 0;
    let mut elements = 0;

    // Split by combinators
    let cleaned = selector
        .replace('>', " ")
        .replace('+', " ")
        .replace('~', " ");
    let parts: Vec<&str> = cleaned.split_whitespace().collect();

    for part in parts {
        ids += part.matches('#').count();
        classes_attrs += part.matches('.').count();
        classes_attrs += part.matches('[').count();

        // Check for element type
        let clean_part = part
            .replace(|c: char| c == '#' || c == '.' || c == '[' || c == ']', " ");
        let first_word = clean_part.split_whitespace().next().unwrap_or("");
        if !first_word.is_empty() && first_word != "*" && first_word.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
            elements += 1;
        }
    }

    (ids, classes_attrs, elements)
}

#[pymethods]
impl CssMatcher {
    #[new]
    fn new() -> Self {
        CssMatcher {
            rules: Vec::new(),
            tag_index: HashMap::new(),
            id_index: HashMap::new(),
            class_index: HashMap::new(),
            universal_rules: Vec::new(),
        }
    }

    /// Add a CSS rule (called once per rule when parsing CSS)
    fn add_rule(&mut self, selector: &str, rule_index: usize) {
        let specificity = calculate_specificity(selector);

        // Parse selector into parts (split by whitespace for descendant combinator)
        let parts: Vec<ParsedSelector> = selector
            .split_whitespace()
            .map(|s| parse_simple_selector(s))
            .collect();

        if parts.is_empty() {
            return;
        }

        let rule = CssRule {
            parts,
            index: rule_index,
            specificity,
        };

        let rule_idx = self.rules.len();

        // Index by the last part (the one that must match the element)
        let last_part = &rule.parts[rule.parts.len() - 1];

        let mut indexed = false;

        // Index by ID (most specific)
        if let Some(ref id) = last_part.id {
            self.id_index
                .entry(id.clone())
                .or_insert_with(Vec::new)
                .push(rule_idx);
            indexed = true;
        }

        // Index by tag
        if let Some(ref tag) = last_part.tag {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(rule_idx);
            indexed = true;
        }

        // Index by first class
        if !last_part.classes.is_empty() {
            self.class_index
                .entry(last_part.classes[0].clone())
                .or_insert_with(Vec::new)
                .push(rule_idx);
            indexed = true;
        }

        // If no specific index, add to universal rules
        if !indexed {
            self.universal_rules.push(rule_idx);
        }

        self.rules.push(rule);
    }

    /// Get the number of rules
    fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Match an element against all rules and return matching rule indices
    /// sorted by specificity (lowest first, so later rules override)
    ///
    /// Args:
    ///   elem_tag: Element tag name (stripped of namespace)
    ///   elem_id: Element id attribute (empty string if none)
    ///   elem_classes: List of element classes
    ///   elem_attrs: Dict of element attributes
    ///   ancestors: List of (tag, id, classes) tuples for ancestors (closest first)
    ///
    /// Returns:
    ///   List of matching rule indices sorted by (specificity, order)
    fn match_element(
        &self,
        elem_tag: &str,
        elem_id: &str,
        elem_classes: Vec<String>,
        elem_attrs: HashMap<String, String>,
        ancestors: Vec<(String, String, Vec<String>)>,
    ) -> Vec<usize> {
        let mut matches: Vec<(usize, usize, usize, usize)> = Vec::new(); // (ids, classes, elements, index)

        // Collect candidate rules from indices
        let mut candidates: Vec<usize> = Vec::new();

        // Check ID index
        if !elem_id.is_empty() {
            if let Some(rules) = self.id_index.get(elem_id) {
                candidates.extend(rules);
            }
        }

        // Check tag index
        if let Some(rules) = self.tag_index.get(elem_tag) {
            candidates.extend(rules);
        }

        // Check class indices
        for class in &elem_classes {
            if let Some(rules) = self.class_index.get(class) {
                candidates.extend(rules);
            }
        }

        // Add universal rules
        candidates.extend(&self.universal_rules);

        // Deduplicate candidates
        candidates.sort_unstable();
        candidates.dedup();

        // Check each candidate rule
        for &rule_idx in &candidates {
            let rule = &self.rules[rule_idx];

            if self.rule_matches(rule, elem_tag, elem_id, &elem_classes, &elem_attrs, &ancestors) {
                matches.push((
                    rule.specificity.0,
                    rule.specificity.1,
                    rule.specificity.2,
                    rule.index,
                ));
            }
        }

        // Sort by specificity then order
        matches.sort_unstable();

        // Return just the indices
        matches.into_iter().map(|(_, _, _, idx)| idx).collect()
    }
}

impl CssMatcher {
    /// Check if a rule matches an element
    fn rule_matches(
        &self,
        rule: &CssRule,
        elem_tag: &str,
        elem_id: &str,
        elem_classes: &[String],
        elem_attrs: &HashMap<String, String>,
        ancestors: &[(String, String, Vec<String>)],
    ) -> bool {
        // Last part must match current element
        let last_part = &rule.parts[rule.parts.len() - 1];
        if !self.selector_matches(last_part, elem_tag, elem_id, elem_classes, elem_attrs) {
            return false;
        }

        if rule.parts.len() == 1 {
            return true;
        }

        // Check ancestor chain for remaining parts (right to left)
        let remaining_parts = &rule.parts[..rule.parts.len() - 1];
        let mut ancestor_idx = 0;

        for part in remaining_parts.iter().rev() {
            let mut matched = false;
            while ancestor_idx < ancestors.len() {
                let (anc_tag, anc_id, anc_classes) = &ancestors[ancestor_idx];
                // For ancestors, we don't have full attrs, so pass empty map
                let empty_attrs = HashMap::new();
                if self.selector_matches(part, anc_tag, anc_id, anc_classes, &empty_attrs) {
                    matched = true;
                    ancestor_idx += 1;
                    break;
                }
                ancestor_idx += 1;
            }

            if !matched {
                return false;
            }
        }

        true
    }

    /// Check if a simple selector matches element data
    #[inline]
    fn selector_matches(
        &self,
        selector: &ParsedSelector,
        tag: &str,
        id: &str,
        classes: &[String],
        attrs: &HashMap<String, String>,
    ) -> bool {
        // Check tag first (most common filter)
        if let Some(ref required_tag) = selector.tag {
            if required_tag != tag {
                return false;
            }
        }

        // Check ID (very fast rejection)
        if let Some(ref required_id) = selector.id {
            if required_id != id {
                return false;
            }
        }

        // Check classes
        for required_class in &selector.classes {
            if !classes.contains(required_class) {
                return false;
            }
        }

        // Check attributes (least common)
        for (attr_name, attr_val) in &selector.attrs {
            match attr_val {
                Some(val) => {
                    if attrs.get(attr_name) != Some(val) {
                        return false;
                    }
                }
                None => {
                    if !attrs.contains_key(attr_name) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// Register CSS module functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CssMatcher>()?;
    Ok(())
}
