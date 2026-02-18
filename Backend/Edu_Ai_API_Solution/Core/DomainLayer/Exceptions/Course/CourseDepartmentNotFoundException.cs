using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Course
{
    public sealed class CourseDepartmentNotFoundException(int courseID , int DepartmentID) : NotFoundException($"this course with id : {courseID} is not assigned to department with id : {DepartmentID}")
    {
    }
}
