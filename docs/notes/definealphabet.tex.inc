\ExplSyntaxOn

\NewDocumentCommand{\definealphabet}{smmm}
 {% #1 = star for lowercase , #2 = prefix, #3 = suffix, #4 = math alphabet
  \IfBooleanTF { #1 }
   {
    \physicsstudent_definealphabet:nnnnn { #2 } { #3 } { `a } { `z } { #4 }
   }
   {
    \physicsstudent_definealphabet:nnnnn { #2 } { #3 } { `A } { `Z } { #4 }
   }
 }

\msg_new:nnn { physicsstudent/definealphabet } { existing }
 {
  Command ~ '\exp_not:c { #1 #2 #3 }' ~ already ~ defined
 }

\cs_new_protected:Nn \physicsstudent_definealphabet:nnnnn
 {
  \int_step_inline:nnn { #3 } { #4 }
   {
    \__physicsstudent_definealphabet:ennn { \char_generate:nn { ##1 } { 12 } } { #1 } { #2 } { #5 }
   }
 }

\cs_new_protected:Nn \__physicsstudent_definealphabet:nnnn
 {
  \cs_if_exist:cTF { #2 #1 #3 }
   {
    \msg_warning:nnnnn { physicsstudent/definealphabet } { existing } { #2 } { #1 } { #3 }
   }
   {
    \cs_new_protected:cpn { #2 #1 #3 } { #4 { #1 } }
   }
 }
\cs_generate_variant:Nn \__physicsstudent_definealphabet:nnnn { e }

\ExplSyntaxOff

\let\Bbb\relax