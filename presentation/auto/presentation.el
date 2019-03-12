(TeX-add-style-hook
 "presentation"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "10pt" "usenames" "dvipsnames" "table")))
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "graphicx"
    "array"
    "fix-cm"
    "colortbl")
   (TeX-add-symbols
    '("heading" 1)
    "g"
    "rcell"
    "gcell"
    "bcell"
    "light"
    "dark"
    "lightcell"
    "midcell"
    "darkcell")
   (LaTeX-add-labels
    "sec:motivation"))
 :latex)

