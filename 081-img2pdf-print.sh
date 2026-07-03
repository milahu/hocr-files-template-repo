#!/usr/bin/env bash

set -eux

pdf=080-img2pdf.pdf

# num_pages=$(pdfinfo $pdf | grep "^Pages:" | awk '{ print $2 }')

# print_two_pages_per_page=false
print_two_pages_per_page=true

if $print_two_pages_per_page; then

    # print two pages per page

    # prepend empty page to preserve the original book print layout:
    # even pages on the left, odd pages on the right
    # example page_size: 595.276 x 841.89 pts
    page_size=$(pdfinfo $pdf | grep "^Page size:" | sed -E 's/^Page size:\s+//; s/\((A4|A5)\)$//')
    echo "page_size: ${page_size@Q}"
    magick xc:none -page "$page_size" blank-page.pdf
    pdf2=081-img2pdf-print.with-blank-page.pdf
    pdftk blank-page.pdf $pdf cat output $pdf2

    exec lp -o sides=two-sided-short-edge -o media=A4 -o print-quality=best -o number-up=2 "$@" $pdf2

fi

# else: print one page per page
exec lp -o sides=two-sided-long-edge -o media=A4 -o print-quality=best "$@" $pdf
