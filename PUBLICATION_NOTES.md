# Items to complete at publication

This file records publication choices that cannot be fabricated during an offline preflight.

Status (18 October 2026): version 1.1.0 is a new version of the same Zenodo record (version DOI 10.5281/zenodo.23440693); the plan of the balanced-view reference was registered on OSF (https://osf.io/mx3z9) after training, because OSF was unavailable for scheduled maintenance; its locked files are unchanged.

Status (10 October 2026): items 1-2 done (GitHub https://github.com/lekhai0123/skin31-audit; reserved Zenodo DOI 10.5281/zenodo.23268495, version 1.0.0). Item 6: osf.io/hn7yk, pe8q6 and 5ksja are public; osf.io/643ht was withdrawn by mistake and replaced by osf.io/jzsr9 with byte-identical files (checked against prereg/registered/derm7pt).

1. Confirm the public GitHub owner/repository URL and the release version/tag. Add the verified URL as `repository-code` in `CITATION.cff` and as a link in `README.md`.
2. Reserve or obtain the Zenodo DOI, add the version-specific DOI to `CITATION.cff`, and link both GitHub and the deposited archive from the paper and cover letter. Do not use the eventual article DOI as the repository DOI. Use a release DOI in the paper so that the cited archive is fixed.
3. Check author name order, ORCID and affiliation against the manuscript. The metadata says `family-names: Pham`, `given-names: Trong Huynh`. The README now uses the full manuscript author names to avoid ambiguous initials.
4. Update the preferred article citation's publication status only when accurate. The candidate currently uses the CFF-supported `status: in-preparation`; change it to `submitted` when submission actually occurs. Article status is not evidence that the repository has been deposited or the article accepted.
5. Choose Zenodo metadata appropriate to a mixed software/metadata archive. Code is MIT; source-derived lists and outputs have the terms/notices in `LICENSE-DATA`. Do not present all non-code files as MIT. Derm7pt's downloaded source README states CC BY-NC-ND 4.0 (the previous repository draft incorrectly said CC BY-NC-SA). The package contains no images. Confirm the licensing notice for distributed reference metadata and author-generated numeric outputs; a preflight cannot determine their legal classification from their file formats.
6. Verify each OSF record is accessible without login and the publicly archived files match the local registered bundles in `prereg/registered`. Local hash validation alone cannot establish public access or prospective registration timestamps.
7. Create the reviewed archive from this release folder, exclude local data, weights and temporary runs, and verify `SHA256SUMS_release.txt` after transfer. A GitHub/Zenodo deposit must refer to the same final version used by the paper.

No remote repository was created, no DOI was minted, and no upload or publication was performed by this preflight.

Official references:

- [Citation File Format documentation](https://citation-file-format.github.io/)
- [GitHub file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Zenodo documentation](https://help.zenodo.org/)
