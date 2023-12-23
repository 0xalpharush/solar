use super::{
    Diagnostic, DiagnosticBuilder, DiagnosticMessage, DynEmitter, EmissionGuarantee,
    ErrorGuaranteed, FatalAbort, Level,
};
use rsolc_data_structures::{map::FxHashSet, sync::Lock};

/// A handler deals with errors and other compiler output.
/// Certain errors (fatal, bug, unimpl) may cause immediate exit,
/// others log errors for later reporting.
pub struct DiagCtxt {
    inner: Lock<DiagCtxtInner>,
}

struct DiagCtxtInner {
    emitter: Box<DynEmitter>,

    /// The number of errors that have been emitted, including duplicates.
    ///
    /// This is not necessarily the count that's reported to the user once
    /// compilation ends.
    err_count: usize,
    deduplicated_err_count: usize,
    warn_count: usize,
    /// The warning count, used for a recap upon finishing
    deduplicated_warn_count: usize,

    /// This set contains a hash of every diagnostic that has been emitted by this `DiagCtxt`.
    /// These hashes are used to avoid emitting the same error twice.
    emitted_diagnostics: FxHashSet<u64>,

    can_emit_warnings: bool,
}

impl DiagCtxt {
    /// Creates a new `DiagCtxt` with the given diagnostics emitter.
    pub fn new(emitter: Box<DynEmitter>) -> Self {
        Self {
            inner: Lock::new(DiagCtxtInner {
                emitter,
                err_count: 0,
                deduplicated_err_count: 0,
                warn_count: 0,
                deduplicated_warn_count: 0,
                emitted_diagnostics: FxHashSet::default(),
                can_emit_warnings: true,
            }),
        }
    }

    /// Disables emitting warnings.
    pub fn disable_warnings(mut self) -> Self {
        self.inner.get_mut().can_emit_warnings = false;
        self
    }

    /// Emits the given diagnostic with this context.
    #[inline]
    pub fn emit_diagnostic(&self, mut diagnostic: Diagnostic) -> Option<ErrorGuaranteed> {
        self.emit_diagnostic_without_consuming(&mut diagnostic)
    }

    /// Emits the given diagnostic with this context, without consuming the diagnostic.
    ///
    /// **Note:** This function is intended to be used only internally in `DiagnosticBuilder`.
    /// Use [`emit_diagnostic`](Self::emit_diagnostic) instead.
    pub(super) fn emit_diagnostic_without_consuming(
        &self,
        diagnostic: &mut Diagnostic,
    ) -> Option<ErrorGuaranteed> {
        self.inner.lock().emit_diagnostic_without_consuming(diagnostic)
    }
}

/// Diagnostic constructors.
///
/// Note that methods returning a [`DiagnosticBuilder`] must also marked with `#[track_caller]`.
impl DiagCtxt {
    /// Creates a builder at the given `level` with the given `message`.
    #[track_caller]
    pub fn diag<G: EmissionGuarantee>(
        &self,
        level: Level,
        message: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, G> {
        DiagnosticBuilder::new(self, level, message)
    }

    /// Creates a builder at the `Fatal` level with the given `message`.
    #[track_caller]
    pub fn fatal(
        &self,
        message: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, FatalAbort> {
        self.diag(Level::Fatal, message)
    }

    /// Creates a builder at the `Error` level with the given `message`.
    #[track_caller]
    pub fn err(
        &self,
        message: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        self.diag(Level::Error, message)
    }

    /// Creates a builder at the `Warning` level with the given `message`.
    ///
    /// Attempting to `.emit()` the builder will only emit if `can_emit_warnings` is `true`.
    #[track_caller]
    pub fn warn(&self, message: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        self.diag(Level::Warning, message)
    }

    /// Creates a builder at the `Help` level with the given `message`.
    #[track_caller]
    pub fn help(&self, message: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        self.diag(Level::Help, message)
    }

    /// Creates a builder at the `Note` level with the given `message`.
    #[track_caller]
    pub fn note(&self, message: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        self.diag(Level::Note, message)
    }
}

impl DiagCtxtInner {
    fn emit_diagnostic_without_consuming(
        &mut self,
        diagnostic: &mut Diagnostic,
    ) -> Option<ErrorGuaranteed> {
        if diagnostic.level == Level::Warning && !self.can_emit_warnings {
            return None;
        }

        if diagnostic.level == Level::Allow {
            return None;
        }

        let already_emitted = self.insert_diagnostic(diagnostic);
        if !already_emitted {
            // Remove duplicate `Once*` subdiagnostics.
            diagnostic.children.retain_mut(|sub| {
                if !matches!(sub.level, Level::OnceNote | Level::OnceHelp) {
                    return true;
                }
                let sub_already_emitted = self.insert_diagnostic(sub);
                !sub_already_emitted
            });

            // if already_emitted {
            //     diagnostic.note(
            //         "duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`",
            //     );
            // }

            self.emitter.emit_diagnostic(diagnostic);
            if diagnostic.is_error() {
                self.deduplicated_err_count += 1;
            } else if diagnostic.level == Level::Warning {
                self.deduplicated_warn_count += 1;
            }
        }

        if diagnostic.is_error() {
            self.bump_err_count();
            Some(ErrorGuaranteed(()))
        } else {
            self.bump_warn_count();
            None
        }
    }

    /// Inserts the given diagnostic into the set of emitted diagnostics.
    /// Returns `true` if the diagnostic was already emitted.
    fn insert_diagnostic<H: std::hash::Hash>(&mut self, diag: &H) -> bool {
        let hash = rsolc_data_structures::map::ahash::RandomState::new().hash_one(diag);
        !self.emitted_diagnostics.insert(hash)
    }

    fn bump_err_count(&mut self) {
        self.err_count += 1;
        // self.panic_if_treat_err_as_bug();
    }

    fn bump_warn_count(&mut self) {
        self.warn_count += 1;
    }
}