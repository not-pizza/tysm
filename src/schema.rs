use std::collections::HashSet;

use schemars::transform::{transform_subschemas, Transform};
use schemars::Schema;
use serde_json::Value;

pub struct OpenAiTransform;

impl Transform for OpenAiTransform {
    fn transform(&mut self, schema: &mut Schema) {
        if let Some(obj) = schema.as_object_mut() {
            if obj.get("$ref").is_none() {
                obj.insert("additionalProperties".to_string(), Value::Bool(false));
                obj.remove("format");
                obj.remove("minimum");
                obj.remove("maximum");

                // get all the items under "properties"
                let properties = obj
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .map(|obj| {
                        obj.keys()
                            .map(|k| Value::String(k.to_string()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<HashSet<_>>();

                // get the "required" array
                let required = obj
                    .get("required")
                    .and_then(|r| r.as_array())
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<HashSet<_>>();

                if properties != required {
                    // add all the properties to the "required" array
                    let mut required_vec: Vec<Value> = properties.into_iter().collect();
                    // Sort for deterministic ordering
                    required_vec
                        .sort_by(|a, b| a.as_str().unwrap_or("").cmp(b.as_str().unwrap_or("")));
                    obj.insert("required".to_string(), Value::Array(required_vec));
                } else {
                    // Even if properties == required, we need to ensure the array is sorted
                    if let Some(Value::Array(required_array)) = obj.get_mut("required") {
                        required_array
                            .sort_by(|a, b| a.as_str().unwrap_or("").cmp(b.as_str().unwrap_or("")));
                    }
                }
            }

            // oneOf keys need to be replaced with anyOf
            if let Some(one_of) = obj.get("oneOf") {
                obj.insert("anyOf".to_string(), one_of.clone());
            }
            obj.remove("oneOf");
        }
        transform_subschemas(self, schema);
    }
}
