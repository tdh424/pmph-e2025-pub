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

Jóhann and <Yet-Unknown-TA> will be (pre-)grading your first 4 assignments, will be patrolling the online (Absalon) discussion forums, and will assist with lab sessions.

### Course Tracks and Resources

All lectures and lab sessions will be delivered in English.  The
assignments and projects will be posted in English, and while you can
chose to hand in solutions in either English or Danish, English is
preferred. All course material except for the hardware book is distributed
via this GitHub page. **Note: assignment handin is still on Absalon!**

* **The hardware track** of the course covers (lecture) topics related to processor, memory and interconnect design, including cache coherency, which are selected from the book [Parallel Computer Organization and Design, by Michel Dubois, Murali Annavaram and Per Stenstrom,  ISBN 978-521-88675-8. Cambridge University Press, 2012](https://www.cambridge.org/dk/academic/subjects/engineering/computer-engineering/parallel-computer-organization-and-design?format=HB&isbn=9780521886758). The book is available at the local bookstore (biocenter). It is not mandatory to buy it---Cosmin thinks that it is possible to understand the material from the lecture slides, which are detailed enough---but also note that lecture notes are not provided for the hardware track, because of copyright issues.

* **The software track** covers (lecture) topics related to parallel-programming models and recipes to recognize and optimize parallelism and locality of reference.  It demonstrates that compiler optimizations are essential to fully utilizing hardware, and that some optimizations can be implemented both in hardware and software, but with different pro and cons.   [The lecture notes are available here](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf), and additional (facultative) reading material (papers) will be linked with individual lectures; see Course Schedule Section below.

* **The lab track** teaches GPGPU hardware specifics and programming in Futhark, CUDA, and OpenMP. The intent is that the lab track applies in practice some of the parallel programming principles and optimizations techniques discussed in the software tracks. It is also intended to provide help for the weekly assignment, group project, etc.

