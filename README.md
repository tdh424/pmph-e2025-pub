# Programming Massively Parallel Hardware (PMPH), Block 1, 2025

### We are grateful to Nvidia for awarding a teaching grant (for the PMPH and DPP courses) that consists of two A100 GPUs. These are now accessible on the Hendrix GPU cluster.

## Course Structure

PMPH is structured to have four hours of (physical) lectures
and four hours of (physical) labs per week; potentially we
will have no lectures in the last few weeks of the course, so you
can concentrate on project work (to be announced).

[Course Catalog Web Page](https://kurser.ku.dk/course/ndak14008u/2025-2026)

### Lectures:

* Monday    13:00 - 15:00 (aud - AUD 01 AKB, Universitetsparken 13)
* Wednesday 10:00 - 12:00 (aud - Aud 01 HCØ, i.e., Universitetsparken 5, HCØ)

### Labs:

* Monday    15:00 - 17:00 (aud - AUD 01 AKB, Universitetsparken 13)
* Wednesday 13:00 - 15:00 (aud - Aud 03, Universitetsparken 5, HCØ)

### Flexible Schedule on Wednesday

For *Wednesdays*, we have also reserved room (aud - Aud 03, Universitetsparken 5, HCØ)
in continuation, from 15:00 -- 17:00, so that we can stay over if necessary.


### Physical Attendence to Lectures and Labs

The current plan is that everybody will have a physical place
at the lectures and labs. Unless we are forced to move to virtual
teaching (very unlikely), the lectures and labs will not be recorded,
so please plan to attend. If there is a strong request, we may stream
the lectures, but without providing any guarantees as to the quality
of streaming.

### Evaluation

Throughout the course, you will hand in six assignments, which will sum
up to 100 points. Final grades are computed according to the following scheme:
* $\geq$ 90 translates to grade 12
* $\geq$ 80 translates to grade 10
* $\geq$ 66 translates to grade  7
* $\geq$ 56 translates to grade  4
* $\geq$ 50 translates to grade  2
* $<$ 50 means fails, i.e., grade 0

All assignments are individual in the sense that each student has to fully 
understand all aspects of the solution (which includes the submitted code)
and the report should include personal reflections on the handed-in solution.
However, some assignments can be solved by a group of students, as detailed
in the itemized list below, and the report should document the name of the
student who wrote a specific section/part of the report. Of course, all
group members should have roughly equal contributions to the solution and
report writing.
If there is disagreement within a group, you may duplicate disputed sections
to have different content, while documenting the name of the authors.

* Assignment 1: 10 points, fully individual solutions and report.

* Assignment 2: 15 points, fully individual solutions and report.

* Assignment 3 (10 points) and 4 (15 points) are to be solved in groups of two.
  Both have the same hand-in and hand-out date, as detailed later, since they
  have a common theme: optimizing locality of reference.

* Assignment 5 (25 points) and 6 (25 points) are to be solved in groups of up to
    3 or 4 students, roughly during the last month of the course.
    They essentially consists of a larger project, in which Assignment 5 refers to
    submitting the code solution, and Assignment 6 refers to submitting the report.
    This intends to stress that the quality of the report will be reflected in
    the final grade. Meaning: if the report is poor, you will loose points.
    It also stands to reason that the quality of the report is tightly
    related to the quality of the solution (implementation), i.e., if the solution
    is not up to the standards, the report is bound to suffer as well.

* There will be no oral presentation of the "group" project (i.e., Assignment 5 & 6),
    because the number of enrolled students has grown and it is not practical anymore.

### Resubmission of Assignments

Assignments 1-4 can be resubmitted *once*, within one week since the time when
you have received feedback for each of them. If you did not get the maximal number
of points, you are allowed to resubmit, only if you wish to do so (not mandatory).

For Assignments 5 and 6, no resubmission is possible, since the deadlines are
just before exam week (Assignment 5) and within the exam week (Assignment 6).

For more information on weekly assignments and (re-)submission, please see [this
section](#weekly-assignments).

**The submission and re-submission of all Assignments is on Absalon. Please do NOT submit solutions on this Github Repo.**

### Teachers

Teachers: 

* **Cosmin Oancea** ([cosmin.oancea@diku.dk](mailto:cosmin.oancea@diku.dk)). Cosmin is typically holding most lectures and labs of PMPH, but this year he might be in parental leave for the first week(s) of the course.
* **Nikolaj Hinnerskov** [nihi@di.ku.dk](mailto:nihi@di.ku.dk). Nikolaj might have to hold the lectures and labs of the first week. 
* **Martin Elsman**      [mael@di.ku.dk](mailto:mael@di.ku.dk). If necessary, Martin and Nikolaj will collaborate to cover the lectures and labs of the second week of the course.

Teaching assistants (TAs):
* **Jóhann Utne** [johann.utne@di.ku.dk](mailto:johann.utne@di.ku.dk)
* ...

Cosmin typically conducts the lectures and lab sessions, but this year he might be in parental leave for the first two week(s) of the course. If so, the plan is that Nikolaj will hold the lectures and labs of the first week of the course, and, if necessary, Nikolaj and Martin will collaborate to cover the second week. 

Jóhann and Yet-Unknown-TA will be (pre-)grading your first 4 assignments, will be patrolling the online (Absalon) discussion forums, and will assist with lab sessions.

### Course Tracks and Resources

All lectures and lab sessions will be delivered in English.  The
assignments and projects will be posted in English, and while you can
chose to hand in solutions in either English or Danish, English is
preferred. All course material except for the hardware book is distributed
via this GitHub page. **Note: assignment handin is still on Absalon!**

* **The hardware track** of the course covers (lecture) topics related to processor, memory and interconnect design, including cache coherency, which are selected from the book [Parallel Computer Organization and Design, by Michel Dubois, Murali Annavaram and Per Stenstrom,  ISBN 978-521-88675-8. Cambridge University Press, 2012](https://www.cambridge.org/dk/academic/subjects/engineering/computer-engineering/parallel-computer-organization-and-design?format=HB&isbn=9780521886758). The book is available at the local bookstore (biocenter). It is not mandatory to buy it---Cosmin thinks that it is possible to understand the material from the lecture slides, which are detailed enough---but also note that lecture notes are not provided for the hardware track, because of copyright issues.

* **The software track** covers (lecture) topics related to parallel-programming models and recipes to recognize and optimize parallelism and locality of reference.  It demonstrates that compiler optimizations are essential to fully utilizing hardware, and that some optimizations can be implemented both in hardware and software, but with different pro and cons.  **The lecture notes are available here** [pdf](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf), [bib](facultative-material/lecture-notes.bib). Additional (facultative) reading material (papers) will be linked with individual lectures; see Course Schedule Section below.

* **The lab track** teaches GPGPU hardware specifics and programming in Futhark, CUDA, and OpenMP. The intent is that the lab track applies in practice some of the parallel programming principles and optimizations techniques discussed in the software tracks. It is also intended to provide help for the weekly assignment, group project, etc.

## Course Schedule

This course schedule is tentative and will be updated as we go along. 
The links will become functional as we get near to the corresponding date.

The lab sessions are aimed at providing help for the weeklies and
group project.  Do not assume you can solve them without attending
the lab sessions.

| Date | Time | Topic | Material |
| --- | --- | --- | --- |
| 01/09 | 13:00-15:00 | [Intro, Hardware Trends and List Homomorphisms (LH - SFT)](slides/L1-Intro-Org-LH.pdf), Chapters 1 and 2 in [Lecture Notes](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf) | Facultative material: [Sergei Gorlatch, "Systematic Extraction and Implementation of Divide-and-Conquer Parallelism"](facultative-material/List-Hom/GorlatchDivAndConq.pdf);  [Richard S. Bird, "An Introduction to the Theory of Lists"](facultative-material/List-Hom/BirdThofLists.pdf); [Jeremy Gibons, "The third homomorphism theorem"](facultative-material/List-Hom/GibonsThirdTheorem.pdf) |
| 01/09 | 15:00-17:00 | [Gentle Intro to CUDA](slides/Lab1-CudaIntro.pdf) | [helper CUDA code](HelperCode/Lab-1-Cuda); as facultative material you may consult Cuda tutorials, for example [a very simple one is this one](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) and [a more comprehensive one is this one](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
| 03/09 | 10:00-12:00 | [List Homomorphism (LH) & Parallel Basic Blocks (SFT)](slides/L2-Flatenning.pdf), Chapters 2 and 3 in [Lecture Notes](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf) | Facultative material: [Various papers related to SCAN and flattening, but which are not very accessible to students](facultative-material/Flattening) |
| 03/09 | 13:00-15:00 | Lab: Futhark programming, First Weekly | [Futhark code related to the LH lecture](HelperCode/Lect-1-LH) and as well [Futhark code related to flattening](HelperCode/Lect-2-Flat). As facultative but useful material: [Parallel Programming in Futhark](https://futhark-book.readthedocs.io/en/latest/), sections 1-4, |
| 03/09 | some time   | [**Assignment 1 handout**](weeklies/weekly-1/) | |
| 08/09 | 13:00-15:00 | [Parallel Basic Block & Flattening Nested Parallelism (SFT)](slides/L2-Flatenning.pdf) | chapters 3 and 4 in [Lecture Notes](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf) |
| 08/09 | 15:00-17:00 | Lab: [Fun Quiz](slides/Lab-fun-quiz.pdf); | help with weekly |
| 10/09 | 10:00-12:00 | [In-Order Pipelines (HWD)](slides/L3-InOrderPipe.pdf)| Chapter 3 of "Parallel Computer Organization and Design" Book |
| 10/09 | 13:00-15:00 | Lab: [Reduce and Scan in Cuda](slides/Lab2-RedScan.pdf) | discussing second weekly, helping with the first |
| 10/09 | some time   | [**Assignment 2 handout**](weeklies/weekly-2/) | |
| 15/09 | 13:00-15:00 | [In-Order Pipelines (HWD)](slides/L3-InOrderPipe.pdf), [Optimizing ILP, VLIW Architectures (SFT-HWD)](slides/L4-VLIW.pdf) | Chapter 3 of "Parallel Computer Organization and Design" Book |
| 15/09 | 15:00-17:00 | Lab: [GPU hardware: three important design choices.](slides/Lab2-GPU-HWD.pdf) | helping with weeklies |
| 17/09 | 10:00-12:00 | [Dependency Analysis of Imperative Loops](slides/L5-LoopParI.pdf) | Chapter 5 of lecture Notes |
| 17/09 | 13:00-15:00 |  | helping with the first two weekly assignments.
| 17/09 |  | No new weekly assignment this week; the third will be published next week | |
| 22/09 | 13:00-15:00 | [Demonstrating Simple Techniques for Optimizing Locality](slides/L6-locality.pdf) | Chapter 5 and 6 of Lecture Notes |
| 22/09 | 15:00-17:00 | [**Assignment 3+4 handout**](weeklies/weekly-3-4/) | helping with the weekly assignments. |
| 24/09 | 10:00-12:00 | [Optimizing Locality Continuation](slides/L6-locality.pdf); [Optimizing Locality same idea in other words: Nearest Neighbor, and again Matrix Multiplication and Transposition](slides/L5-LoopParI.pdf) | Chapters 5 and 6 of lecture Notes |
| 24/09 | 13:00-15:00 | Lab: discussing the third assignment | helping with the weekly assignments.
| 29/09 | 13:00-15:00 | helping with assignments | due to still not having my voice back
| 29/09 | 15:00-17:00 | Lab: [**Presenting Possible Group Projects**](group-projects/) | discussing group projects, helping with weekly assignments |
| 01/10 | 10:00-12:00 | [Memory Hierarchy, Bus-Based Coherency Protocols (HWD)](slides/L7-MemIntro.pdf) | Chapter 4 and 5 of "Parallel Computer Organization and Design" Book |
| 01/10 | 13:00-15:00 | Lab: [**Presenting Possible Group Projects**](group-projects/) | helping with weekly assignments, discussing group projects.
| 06/10 | 13:00-15:00 | HWD: [Bus-Based Coherency Protocols](slides/L7-MemIntro.pdf), and [Scalable Coherence Protocols](slides/L8-Interconnect.pdf) | Chapters 5 and 6 of "Parallel Computer Organization and Design" Book |
| 06/10 | 15:00-17:00 | Lab: helping with weekly assignments and project |  |
| 08/10 | 10:00-12:00 | HWD: [Scalable Coherence Protocols, Scalable Interconect (HWD)](slides/L8-Interconnect.pdf); if time permits [Exercises related to cache coherency and interconnect](hwd-exercises/hwd-coherence-in-exercises.pdf)| Chapters 5 and 6 of "Parallel Computer Organization and Design" Book |
| 08/10 | 13:00-15:00 | | helping with weeklies and project
| 13/10 | 13:00-15:00 | Autumn break (no lecture) | |
| 13/10 | 15:00-17:00 | Autumn break (no lab) | |
| 15/10 | 10:00-12:00 | Autumn break (no lecture) | |
| 15/10 | 13:00-15:00 | Autumn break (no lab) |
| 20/10 | 13:00-15:00 | [Demonstrating by Exercises the Coherency Protocols and Interconnect material](hwd-exercises/hwd-coherence-in-exercises.pdf) | |
| 20/10 | 15:00-17:00 | Lab: Helping with group project and weeklies | |
| 22/10 | 10:00-12:00 | [Inspector-Executor Techniques for Locality Optimizations (SFT)](slides/L9-LocOfRef.pdf) | [Facultative reading: various scientific papers](facultative-material/Opt-Loc-Ref) |
| 22/10 | 13:00-15:00 | Lab: help with group project, weeklies |
| 27/10 | 13:00-15:00 | Lecture: helping with group project and weeklies | you may read Tomasulo Algorithm (HWD) from Chapter 3 of "Parallel Computer Organization and Design" Book; [also on slides](slides/L9-OoOproc.pdf) |
| 27/10 | 15:00-17:00 | Lab: Helping with group project, weeklies | |
| 29/10 | 10:00-12:00 | Lecture: helping with group project | |
| 29/10 | 13:00-15:00 | Lab: help with group project |
| 02/11 |             | Assignment 5 Submission Deadline |
| 05/11 |             | Assignment 6 Submission Deadline | 
