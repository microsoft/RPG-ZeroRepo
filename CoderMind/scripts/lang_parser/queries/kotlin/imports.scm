; cmind-imports.scm — import captures for Kotlin
;
; This file closes the one known gap in fwcd's tags.scm for cmind's needs:
; imports (tags.scm upstream has no import capture at all). It is a
; candidate for upstreaming into fwcd/tree-sitter-kotlin; if accepted,
; this file can be deleted and the capture inherited from upstream.
;
; Grammar: fwcd/tree-sitter-kotlin @ 1852ea17b7f60fb3f9d84e0b1555d56b46b39fb1

(import_header
  (identifier) @import.path) @definition.import

(import_header
  (import_alias
    (type_identifier) @import.alias)) @definition.import

; Unnamed companion objects: tags.scm only captures NAMED companion_object
; nodes. cmind's contract requires every companion to appear as a class
; unit, so capture unnamed ones too (adapter falls back to name "companion").
(companion_object) @definition.class
