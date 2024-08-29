\ExplSyntaxOn

\NewDocumentCommand{\definealphabet}{smmm}
 {% #1 = star for lowercase , #2 = prefix, #3 = suffix, #4 = math alphabet
  \IfBooleanTF { #1 }
   {
    \symbolshortcut_definealphabet:nnnnn { #2 } { #3 } { `a } { `z } { #4 }
   }
   {
    \symbolshortcut_definealphabet:nnnnn { #2 } { #3 } { `A } { `Z } { #4 }
   }
 }

\msg_new:nnn { symbolshortcut/definealphabet } { existing }
 {
  Command ~ '\exp_not:c { #1 #2 #3 }' ~ already ~ defined
 }

\cs_new_protected:Nn \symbolshortcut_definealphabet:nnnnn
 {
  \int_step_inline:nnn { #3 } { #4 }
   {
    \__symbolshortcut_definealphabet:ennn { \char_generate:nn { ##1 } { 12 } } { #1 } { #2 } { #5 }
   }
 }

\cs_new_protected:Nn \__symbolshortcut_definealphabet:nnnn
 {
  \cs_if_exist:cTF { #2 #1 #3 }
   {
    \msg_warning:nnnnn { symbolshortcut/definealphabet } { existing } { #2 } { #1 } { #3 }
   }
   {
    \cs_new_protected:cpn { #2 #1 #3 } { #4 { #1 } }
   }
 }
\cs_generate_variant:Nn \__symbolshortcut_definealphabet:nnnn { e }

\ExplSyntaxOff

\let\Bbb\relax